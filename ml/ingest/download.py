"""
Step 2 — download and extract source distributions (sdists) from PyPI.

An "sdist" is the .tar.gz of a package's actual source code. We need the
source, not the installed wheel, because griffe reads source files.

Try it on one package:

    python ml/ingest/download.py click
"""

import io
import pathlib
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


def list_releases(package: str, last_n: int = 15) -> list[dict]:
    """
    The last N real releases of `package` that ship an sdist, oldest first.

    Skips pre-releases (2.0.0rc1), dev releases, and yanked files —
    none of those represent "what users actually upgraded to".
    """
    data = get_json(f"https://pypi.org/pypi/{package}/json")

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
    return out[-last_n:]


def download_and_extract(url: str, dest: pathlib.Path) -> pathlib.Path | None:
    """
    Download an sdist, extract it into `dest`, and return the folder that
    Python would import the package FROM (not the package folder itself).

    Returns None if the layout can't be worked out — log it and move on.
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    r = httpx.get(url, timeout=120, follow_redirects=True)
    r.raise_for_status()
    blob = io.BytesIO(r.content)

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
    for nested in ("src", "lib"):
        candidate = root / nested
        if candidate.is_dir() and any(candidate.iterdir()):
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
