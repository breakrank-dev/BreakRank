"""
Step 2 — download and extract source distributions (sdists) from PyPI.

An "sdist" is the .tar.gz of a package's actual source code. We need the
source, not the installed wheel, because griffe reads source files.

Try it on one package:

    python ml/ingest/download.py click
"""

import io
import pathlib
import re
import sys
import tarfile
import zipfile

import httpx
from packaging.version import InvalidVersion, Version
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_json(url: str) -> dict:
    """GET a JSON URL, retrying up to 3 times with increasing waits."""
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r.json()


GITHUB = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)",
    re.IGNORECASE,
)

# Keys most likely to hold the real source repo, best first. A project can
# list a dozen URLs and several may point at github — "Changelog" often
# points at a release page, "Funding" at a sponsor page.
REPO_KEYS = ("source", "source code", "repository", "code", "github",
             "homepage", "home")


def find_github_repo(info: dict) -> str | None:
    """Best-effort https://github.com/owner/repo from PyPI metadata.

    Nullable and never raises. This is the input to changelog and
    deprecation mining later — a signal griffe cannot give us, because
    "deprecated since 2.1" lives in prose, not in the API surface. But it
    is a nice-to-have, so it must never be able to fail an ingest.
    """
    urls = dict(info.get("project_urls") or {})
    if info.get("home_page"):
        urls.setdefault("home_page", info["home_page"])

    ordered = sorted(
        urls.items(),
        key=lambda kv: next((i for i, k in enumerate(REPO_KEYS)
                             if k in kv[0].lower().strip()), len(REPO_KEYS)),
    )
    for _, url in ordered:
        m = GITHUB.match(str(url or ""))
        if m:
            owner, repo = m.group(1), m.group(2)
            if repo.lower().endswith(".git"):
                repo = repo[:-4]
            # Skip github's own non-repo paths (/sponsors/x, /orgs/x).
            if owner.lower() in {"sponsors", "orgs", "users", "apps"}:
                continue
            return f"https://github.com/{owner}/{repo}"
    return None


def list_releases_with_meta(package: str, last_n: int = 15) -> tuple[list[dict], dict]:
    """
    Releases plus the package-level metadata, from ONE PyPI request.

    The metadata half feeds Varad's `package` table. Split out rather than
    fetched separately because we already have the JSON in hand — a second
    request per package would be 400 extra round trips for a field that is
    sitting right there in the response.
    """
    data = get_json(f"https://pypi.org/pypi/{package}/json")
    info = data.get("info") or {}
    meta = {"package": package, "github_repo": find_github_repo(info)}

    out = []
    for version, files in data["releases"].items():
        try:
            v = Version(version)
        except InvalidVersion:
            continue  # some ancient packages have unparseable version strings
        if v.is_prerelease or v.is_devrelease:
            continue

        sdist = next(
            (f for f in files if f["packagetype"] == "sdist" and not f.get("yanked")),
            None,
        )
        if sdist:
            out.append(
                {
                    "version": version,
                    "parsed": v,
                    "url": sdist["url"],
                    "uploaded": sdist["upload_time_iso_8601"],
                }
            )

    out.sort(key=lambda r: r["parsed"])  # sort by real version order, not string order
    return out[-last_n:], meta


def list_releases(package: str, last_n: int = 15) -> list[dict]:
    """
    The last N real releases of `package` that ship an sdist, oldest first.

    Skips pre-releases (2.0.0rc1), dev releases, and yanked files —
    none of those represent "what users actually upgraded to".
    """
    return list_releases_with_meta(package, last_n)[0]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _fetch_bytes(url: str) -> bytes:
    """Download one file, retrying on transient network failures.

    This retry is not optional politeness — it changes your data. A 50-package
    run dropped `packaging 25.0` to a one-off connection error, which silently
    removed the whole 25.0 -> 26.0 version pair and two real rows from the
    dataset. Two runs of identical code produced different CSVs. The analysis
    is deterministic; the network is not, and without a retry that noise ends
    up in your training data.
    """
    r = httpx.get(url, timeout=120, follow_redirects=True)
    r.raise_for_status()
    return r.content


def download_and_extract(url: str, dest: pathlib.Path) -> pathlib.Path | None:
    """
    Download an sdist, extract it into `dest`, and return the folder that
    Python would import the package FROM (not the package folder itself).

    Returns None if the layout can't be worked out — log it and move on.
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    blob = io.BytesIO(_fetch_bytes(url))

    if url.endswith(".zip"):
        with zipfile.ZipFile(blob) as zf:
            zf.extractall(dest)
    else:
        # filter="data" refuses tar entries that would write outside dest.
        # Needs Python 3.11.4+ — another reason for the version check.
        with tarfile.open(fileobj=blob, mode="r:*") as tar:
            tar.extractall(dest, filter="data")

    roots = [p for p in dest.iterdir() if p.is_dir()]
    if not roots:
        return None
    root = roots[0]  # sdists unpack into a single folder like click-8.2.0/

    # Three layouts in the wild, and you have to check for all of them:
    #   click-8.2.0/src/click/__init__.py   ->  search path is .../src
    #   PyYAML-6.0.2/lib/yaml/__init__.py   ->  search path is .../lib
    #   six-1.16.0/six.py                   ->  search path is the root itself
    #
    # "Non-empty" is not the test — "contains Python" is. matplotlib ships
    # BOTH: src/ holds its C++ extension sources (_backend_agg.cpp, _path.h,
    # _macosx.m — not one .py file) and lib/ holds the actual library. The
    # old check took src/ because it existed and had files in it, found no
    # Python, and logged matplotlib as "compiled-only or not a Python
    # package". A top-30 package, silently dropped from the dataset by a
    # directory name.
    for nested in ("src", "lib"):
        candidate = root / nested
        if candidate.is_dir() and any(candidate.rglob("*.py")):
            return candidate
    return root


if __name__ == "__main__":
    package = sys.argv[1] if len(sys.argv) > 1 else "click"

    releases = list_releases(package, last_n=10)
    print(f"\nLast {len(releases)} sdist releases of '{package}':\n")
    for r in releases:
        print(f"  {r['version']:<12} uploaded {r['uploaded'][:10]}")

    newest = releases[-1]
    print(f"\nDownloading {package} {newest['version']} ...")
    path = download_and_extract(newest["url"], pathlib.Path(f"data/sdists/{package}/{newest['version']}"))
    print(f"Extracted. Import search path: {path}")
    print(f"Python files found: {len(list(path.rglob('*.py')))}")
