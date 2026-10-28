"""
Week 5's evaluation harness (NOTES §38.6, §38.18): the fixer run on the
ten sample packages, one run per upgrade each meets, with everything it
downloaded, read and produced recorded under data/eval/.

    python scripts/eval_run.py plan
    python scripts/eval_run.py resolve fetch old new run count
    python scripts/eval_run.py draw

Stages, each re-runnable alone (what is already done is kept; --redo
repeats it):
  plan     the runs, from data/eval_sample.csv (the rows marked sample)
           and the bundle: (sample package, changed package, from, to);
           the distinct old and new versions they need
  resolve  PyPI's record of each sample package: the newest final release
           (no pre-, dev- or yanked release), its sdist, else its
           pure-Python wheel; filename, size, sha256, upload time
  fetch    download each archive once, check its sha256, unpack it; the
           tree is read, never built, installed or imported
  old      each (changed package, old version) into data/eval/old/<p>-<v>:
           `pip install --only-binary=:all: --no-deps --target`, a flat
           directory the fixer's --env takes as a site-packages; where no
           wheel exists for this Python, the sdist is unpacked, not built,
           and its package placed beside a hand-written dist-info
  new      the same for each new version, into data/eval/new/, for the
           reader's eyes only (the verdicts need the new signature)
  run      breakrank-fix <changed> <tree> --from --to --env --diff, per
           planned run; stdout and stderr to <sample>/<p>-<from>-<to>.txt,
           the diff beside it; exit code and seconds to the manifest
  count    parse every report into data/eval/counts.csv and print totals
  draw     the calls to read: §38.18's 200 rewritten and 30 withheld,
           seeds 20261028 and 20261029, to data/eval/draw.csv

Nothing here is committed: data/ is ignored. The manifest,
data/eval/manifest.json, records the tool's version and commit, the
interpreters, every download with its digest and every run with its
exit code; the record in NOTES is made from it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fixer.bundle import version_key  # noqa: E402
from fixer.env import normalise  # noqa: E402
from scripts.eval_sample import load_functions  # noqa: E402

try:
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover - packaging is the fixer's dependency
    raise SystemExit("packaging is needed (pip install packaging)")

ROOT = pathlib.Path("data/eval")
SAMPLE = pathlib.Path("data/eval_sample.csv")
BUNDLE = pathlib.Path("fixer/data/breakrank.sqlite")
PYPI = "https://pypi.org/pypi/{}/json"
PYPI_VERSION = "https://pypi.org/pypi/{}/{}/json"
TIMEOUT = 600
REWRITTEN, WITHHELD = 200, 30
SEED_REWRITTEN, SEED_WITHHELD = 20261028, 20261029
DECISIONS = {"rewritten", "withheld", "left to you", "needs nothing"}


class PlanError(Exception):
    """The sample csv and the bundle do not agree; nothing is planned."""


# --------------------------------------------------------------- the plan

def plan(sample: pathlib.Path, bundle: pathlib.Path):
    """(runs, olds, news): runs are (sample package, changed package,
    from, to), sorted; olds and news the distinct (package, version)
    pairs they need."""
    _, upgrades, _ = load_functions(bundle)
    runs = set()
    with sample.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["role"] != "sample":
                continue
            for item in row["detail"].split(";"):
                if not item:
                    continue
                pkg, function = item.split(":", 1)
                ups = upgrades.get((pkg, function))
                if not ups:
                    raise PlanError(
                        f"{row['scanner']} uses {pkg}:{function}, on which "
                        "the bundle has no rewritable row; the csv is not "
                        "this bundle's")
                for frm, to in ups:
                    runs.add((row["scanner"], pkg, frm, to))
    runs = sorted(runs, key=lambda r: (r[0], r[1], version_key(r[2]),
                                       version_key(r[3])))
    olds = sorted({(r[1], r[2]) for r in runs},
                  key=lambda p: (p[0], version_key(p[1])))
    news = sorted({(r[1], r[3]) for r in runs},
                  key=lambda p: (p[0], version_key(p[1])))
    return runs, olds, news


# ----------------------------------------------------------- the releases

def _final(version: str) -> bool:
    try:
        v = Version(version)
    except InvalidVersion:
        return False
    return not (v.is_prerelease or v.is_devrelease)


def choose_release(info: dict):
    """(version, file, kind) for the newest final release with files:
    kind 'sdist', 'wheel' (pure Python, py3-none-any) or None when the
    release has neither; (None, None, None) when nothing is final."""
    versions = [v for v in info.get("releases", {}) if _final(v)]
    for version in sorted(versions, key=version_key, reverse=True):
        files = [f for f in info["releases"][version]
                 if not f.get("yanked", False)]
        if not files:
            continue
        for f in files:
            if f["packagetype"] == "sdist":
                return version, f, "sdist"
        for f in files:
            if f["packagetype"] == "bdist_wheel" and f["filename"].endswith(
                    ("-py3-none-any.whl", "-py2.py3-none-any.whl")):
                return version, f, "wheel"
        return version, None, None
    return None, None, None


def pypi(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download(url: str, dest: pathlib.Path, sha256: str) -> int:
    """The file to dest, its sha256 checked; the size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, dest.open("wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
    return verify(dest, sha256)


def verify(dest: pathlib.Path, sha256: str) -> int:
    """The file's sha256 against PyPI's; a wrong one is deleted and
    refused. The size."""
    digest = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        dest.unlink()
        raise RuntimeError(f"{dest.name}: sha256 {digest.hexdigest()} is not "
                           f"PyPI's {sha256}")
    return dest.stat().st_size


def unpack(archive: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """The archive unpacked under dest; the tree root (the one top-level
    directory when there is one, else dest)."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    elif archive.name.endswith((".zip", ".whl")):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if n.startswith("/") or ".." in pathlib.PurePosixPath(n).parts:
                    raise RuntimeError(f"{archive.name}: unsafe member {n}")
            z.extractall(dest)
    else:
        raise RuntimeError(f"{archive.name}: not an archive this reads")
    tops = [p for p in dest.iterdir()]
    return tops[0] if len(tops) == 1 and tops[0].is_dir() else dest


# -------------------------------------------------------- the old versions

def install_wheel(python: str, pkg: str, version: str,
                  dest: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [python, "-m", "pip", "install", "--only-binary=:all:", "--no-deps",
         "--disable-pip-version-check", "--quiet", "--target", str(dest),
         f"{pkg}=={version}"], capture_output=True, text=True, timeout=900)


NOT_MODULES = {"setup.py", "conftest.py", "versioneer.py", "noxfile.py",
               "tasks.py"}


def find_packages(tree: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
    """Where an sdist keeps its import packages, and their names: the
    first of lib3/, src/, lib/, the root that holds a directory with an
    __init__.py, or failing that a top-level module (a single-module
    distribution, src/typing_extensions.py). At the root, tests, docs,
    examples and the like are left aside; under lib3/, src/ and lib/
    everything is the package's."""
    skip = {"tests", "test", "docs", "doc", "examples", "benchmarks",
            "scripts", "tools", "build", "dist"}
    for sub in ("lib3", "src", "lib", "."):
        base = tree / sub
        if not base.is_dir():
            continue
        names = sorted(p.name for p in base.iterdir()
                       if p.is_dir() and (p / "__init__.py").is_file()
                       and (sub != "." or p.name not in skip)
                       and not p.name.startswith("."))
        modules = sorted(p.stem for p in base.iterdir()
                         if p.is_file() and p.suffix == ".py"
                         and p.name not in NOT_MODULES
                         and not p.name.startswith(("test_", ".")))
        if names:
            return base, names + (modules if sub != "." else [])
        if modules:
            return base, modules
    return tree, []


def place_unpacked(tree: pathlib.Path, pkg: str, version: str,
                   dest: pathlib.Path) -> list[str]:
    """An unpacked sdist's packages copied into dest beside a hand-written
    dist-info (METADATA: name and version; top_level.txt: the packages),
    so that fixer/env.py reads dest as a site-packages. Nothing is run."""
    base, names = find_packages(tree)
    if not names:
        raise RuntimeError(f"{tree}: no package directory or module found")
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        if (base / n).is_dir():
            shutil.copytree(base / n, dest / n, dirs_exist_ok=True)
        else:
            shutil.copy2(base / f"{n}.py", dest / f"{n}.py")
    info = dest / f"{normalise(pkg)}-{version}.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {pkg}\nVersion: {version}\n",
        encoding="utf-8")
    (info / "top_level.txt").write_text("\n".join(names) + "\n",
                                        encoding="utf-8")
    return names


def no_wheel(pip_output: str) -> bool:
    """pip's one message that means there is no wheel for this Python;
    any other failure (the index unreachable, which ends in the same
    line after "(from versions: none)" and retries; no space) is not
    that."""
    if "(from versions: none)" in pip_output or "Retrying" in pip_output:
        return False
    return "No matching distribution found" in pip_output


def obtain(python: str, pkg: str, version: str, dest: pathlib.Path,
           scratch: pathlib.Path) -> dict:
    """The version into dest: from its wheel when pip finds one for this
    Python, else (when pip says there is none) its sdist unpacked and
    placed. What was done, recorded."""
    r = install_wheel(python, pkg, version, dest)
    if r.returncode == 0:
        return {"how": "wheel", "python": python}
    said = (r.stderr or r.stdout).strip()
    pip_said = said.splitlines()[-1:]
    if not no_wheel(said):
        raise RuntimeError(f"{pkg} {version}: pip failed and not for want of "
                           f"a wheel: {' '.join(pip_said)}")
    info = pypi(PYPI_VERSION.format(pkg, version))
    sdists = [f for f in info.get("urls", []) if f["packagetype"] == "sdist"]
    if not sdists:
        raise RuntimeError(f"{pkg} {version}: no wheel for this Python "
                           f"({pip_said}) and no sdist on PyPI")
    f = sdists[0]
    archive = scratch / f["filename"]
    size = download(f["url"], archive, f["digests"]["sha256"])
    tree = unpack(archive, scratch / f"{pkg}-{version}-src")
    names = place_unpacked(tree, pkg, version, dest)
    return {"how": "sdist-unpacked", "pip_said": pip_said,
            "sdist": f["filename"], "size": size,
            "sha256": f["digests"]["sha256"], "packages": names}


# ---------------------------------------------------------------- the runs

def run_one(tool: list[str], run, tree: pathlib.Path, env: pathlib.Path,
            root: pathlib.Path, timeout: int = TIMEOUT, extra=()) -> dict:
    sample, pkg, frm, to = run
    out_dir = root / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pkg}-{frm}-{to}"
    diff = out_dir / f"{stem}.diff"
    cmd = list(tool) + [pkg, str(tree), "--from", frm, "--to", to,
                        "--env", str(env), "--diff", str(diff)] + list(extra)
    start = time.monotonic()
    timed_out = False
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        timed_out, code = True, None

        def text_of(x):
            return x.decode("utf-8", "replace") if isinstance(x, bytes) \
                else (x or "")
        out = text_of(e.stdout)
        err = text_of(e.stderr) + f"\ntimed out after {timeout} s"
    seconds = round(time.monotonic() - start, 1)
    text = out + (("\n[stderr]\n" + err) if err.strip() else "")
    (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    return {"exit": code, "seconds": seconds, "timed_out": timed_out,
            "command": cmd}


_HEAD = re.compile(r"^project .*: (\d+) file\(s\) read, (\d+) mention the "
                   r"import root")
_USE = re.compile(r"^\s+(\d+) line\(s\) use (\d+) of the (\d+) change\(s\), "
                  r"(\d+) hit\(s\): (\d+) resolved, (\d+) referenced, "
                  r"(\d+) unverified\.")
_FOOT = re.compile(r"^R1, pass by name: (\d+) call\(s\) rewritten, (\d+) "
                   r"withheld, (\d+) need nothing\. R2, pin the old default: "
                   r"(\d+) call\(s\) pinned, (\d+) withheld, (\d+) left to "
                   r"you, (\d+) need nothing")
_DIFF = re.compile(r"^diff written to .* \((\d+) file\(s\)\)")
_DECISION = re.compile(r"^\s+-> (rewritten|withheld|left to you|needs nothing)"
                       r"(?:: (.*))?$")
_ENTRY = re.compile(r"^\s+\d+\s+(\S+)\s+used by \d+")
_HIT = re.compile(r"^\s{7}(.+?):(\d+):(\d+)\s+(resolved|referenced|"
                  r"unverified)\s+(\S+)\s+(.*)$")
_RULE = re.compile(r"^\s{39}(R[12].*)$")

COUNT_KEYS = ("files_read", "files_mention", "old_read", "lines",
              "changes_used", "changes", "hits", "resolved", "referenced",
              "unverified", "r1_rewritten", "r1_withheld", "r1_nothing",
              "r2_pinned", "r2_withheld", "r2_left", "r2_nothing",
              "rewritten", "withheld", "left", "nothing", "diff_files")


def parse_report(text: str) -> dict:
    """The report's numbers; None for all of them, and the message in
    'error', when the run stopped before the report or before its
    footer. The decision counts are of CALLS: a call's decision is
    printed under every entry it meets, and `decisions` merges those."""
    c: dict = {k: None for k in COUNT_KEYS}
    c["error"] = ""
    lines = text.splitlines()
    seen_head = False
    for ln in lines:
        m = _HEAD.match(ln)
        if m:
            seen_head = True
            c["files_read"], c["files_mention"] = int(m[1]), int(m[2])
            c["old_read"] = False
            continue
        if "its classes' bases and signatures are known" in ln:
            c["old_read"] = True
        m = _USE.match(ln)
        if m:
            (c["lines"], c["changes_used"], c["changes"], c["hits"],
             c["resolved"], c["referenced"], c["unverified"]) = map(int,
                                                                   m.groups())
            continue
        m = _FOOT.match(ln)
        if m:
            (c["r1_rewritten"], c["r1_withheld"], c["r1_nothing"],
             c["r2_pinned"], c["r2_withheld"], c["r2_left"],
             c["r2_nothing"]) = map(int, m.groups())
            continue
        m = _DIFF.match(ln)
        if m:
            c["diff_files"] = int(m[1])
    tail = [ln for ln in lines if ln.strip()]
    if not seen_head:
        c["error"] = tail[-1] if tail else "no output"
        return c
    if c["r1_rewritten"] is None:
        # the head but no footer: cut off (a timeout, a crash mid-report)
        c["error"] = ("cut off before the footer: "
                      + (tail[-1] if tail else "no output"))
        for k in COUNT_KEYS:
            if k not in ("files_read", "files_mention", "old_read"):
                c[k] = None
        return c
    kinds = collections.Counter(d["kind"] for d in decisions(text))
    c["rewritten"] = kinds["rewritten"]
    c["withheld"] = kinds["withheld"]
    c["left"] = kinds["left to you"]
    c["nothing"] = kinds["needs nothing"]
    if c["diff_files"] is None:
        c["diff_files"] = 0
    # the calls counted against the footer: a call is rewritten by one
    # rule or both, withheld by one or both
    if not (max(c["r1_rewritten"], c["r2_pinned"]) <= c["rewritten"]
            <= c["r1_rewritten"] + c["r2_pinned"]
            and max(c["r1_withheld"], c["r2_withheld"]) <= c["withheld"]
            <= c["r1_withheld"] + c["r2_withheld"]):
        c["error"] = (f"the decision lines do not add up to the footer: "
                      f"{c['rewritten']} rewritten against R1 "
                      f"{c['r1_rewritten']} + R2 {c['r2_pinned']}, "
                      f"{c['withheld']} withheld against R1 {c['r1_withheld']}"
                      f" + R2 {c['r2_withheld']}")
    return c


def decisions(text: str) -> list[dict]:
    """Every decision in a report, one per CALL: the hit line it belongs
    to (file, line, column, confidence, use, code), the kind (rewritten,
    withheld, left to you, needs nothing), the call as rewritten, the
    rule lines under it, and the symbols of every entry it was printed
    under (the report prints a call's decision under each entry whose
    rows the call meets; those are merged here on file, line, column,
    use, kind and call). Sorted by file, line, column, use."""
    found: dict = {}
    symbol = None
    hit = None
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = _ENTRY.match(ln)
        if m:
            symbol = m[1]
            hit = None
            continue
        m = _HIT.match(ln)
        if m:
            hit = {"file": m[1], "line": int(m[2]), "column": int(m[3]),
                   "confidence": m[4], "use": m[5], "code": m[6]}
            continue
        m = _DECISION.match(ln)
        if m and hit is not None:
            rules = []
            for nxt in lines[i + 1:]:
                r = _RULE.match(nxt)
                if not r:
                    break
                rules.append(r[1])
            k = (hit["file"], hit["line"], hit["column"], hit["use"], m[1],
                 m[2] or "", tuple(rules))
            if k in found:
                if symbol not in found[k]["symbols"]:
                    found[k]["symbols"].append(symbol)
            else:
                found[k] = {**hit, "symbol": symbol, "symbols": [symbol],
                            "kind": m[1], "call": m[2] or "", "rules": rules}
            hit = None
    out = list(found.values())
    out.sort(key=lambda d: (d["file"], d["line"], d["column"], d["use"]))
    return out


def count(root: pathlib.Path, runs) -> list[dict]:
    rows = []
    manifest = read_manifest(root)
    for run in runs:
        sample, pkg, frm, to = run
        path = root / sample / f"{pkg}-{frm}-{to}.txt"
        rec = manifest.get("runs", {}).get(key(run), {})
        c = parse_report(path.read_text(encoding="utf-8")) \
            if path.is_file() else {**{k: None for k in COUNT_KEYS},
                                    "error": rec.get("skipped") or "not run"}
        rows.append({"sample": sample, "changed": pkg, "from": frm, "to": to,
                     "exit": rec.get("exit"), "seconds": rec.get("seconds"),
                     **c})
    fields = ["sample", "changed", "from", "to", "exit", "seconds",
              *COUNT_KEYS, "error"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "counts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def totals(rows: list[dict]) -> list[str]:
    """The counts summed per sample package, per changed package and
    overall, as lines; the unverified beside the hits as a count over a
    count; the runs without a report counted apart."""
    lines = []
    for label, field in (("sample package", "sample"),
                         ("changed package", "changed")):
        by = collections.defaultdict(collections.Counter)
        none = collections.Counter()
        for r in rows:
            group = r[field]
            if r["hits"] is None:
                none[group] += 1
                none["ALL"] += 1
                continue
            for k in ("hits", "unverified", "rewritten", "withheld", "left",
                      "nothing", "r1_rewritten", "r2_pinned"):
                by[group][k] += r[k]
                by["ALL"][k] += r[k]
            by[group]["runs"] += 1
            by["ALL"]["runs"] += 1
        lines.append(f"{label:<28} {'runs':>4} {'hits':>5} "
                     f"{'unverified/hits':>15} {'rewritten':>9} "
                     f"{'withheld':>8} {'left':>4} {'nothing':>7} "
                     f"{'R1 rw':>5} {'R2 pin':>6}  no report")
        groups = sorted(set(by) | set(none) - {"ALL"})
        for g in [g for g in groups if g != "ALL"] + ["ALL"]:
            c = by[g]
            lines.append(f"{g:<28} {c['runs']:>4} {c['hits']:>5} "
                         f"{str(c['unverified']) + '/' + str(c['hits']):>15} "
                         f"{c['rewritten']:>9} {c['withheld']:>8} "
                         f"{c['left']:>4} {c['nothing']:>7} "
                         f"{c['r1_rewritten']:>5} {c['r2_pinned']:>6}  "
                         f"{none[g]}")
        lines.append("")
    return lines


# ---------------------------------------------------------------- the draw

def allocate(per_run: dict, budget: int):
    """How many calls each run gives to a draw of `budget`: all of them
    when the budget covers the total; else every run with fewer than m
    gives all, every other at least m (m the largest of 3, 2, 1 whose
    minimums fit), and the rest of the budget in proportion to what each
    run has above m, largest remainder, capped at the run's count. None
    when even one from every run that has one does not fit: the draw
    then takes the budget from all the calls as one list."""
    total = sum(per_run.values())
    if total <= budget:
        return dict(per_run)
    for m in (3, 2, 1):
        base = {k: min(v, m) for k, v in per_run.items()}
        if sum(base.values()) <= budget:
            break
    else:
        return None
    left = budget - sum(base.values())
    above = {k: per_run[k] - base[k] for k in per_run if per_run[k] > base[k]}
    pool = sum(above.values())
    if left and pool:
        exact = {k: left * above[k] / pool for k in above}
        floor = {k: min(int(exact[k]), above[k]) for k in above}
        rest = left - sum(floor.values())
        order = sorted(above, key=lambda k: (-(exact[k] - int(exact[k])), k))
        for k in order:
            if rest <= 0:
                break
            if floor[k] < above[k]:
                floor[k] += 1
                rest -= 1
        for k, n in floor.items():
            base[k] += n
    return base


def draw(pool: dict, budget: int, seed: int):
    """(the calls drawn from each run, how): by `allocate`'s shares, one
    Random(seed) over the runs in sorted order, each run's calls in
    sorted order ("allocated"); or, when one from every run does not
    fit the budget, `budget` calls drawn from all of them as one sorted
    list ("pooled"), which is proportional in expectation."""
    share = allocate({k: len(v) for k, v in pool.items()}, budget)
    rng = random.Random(seed)
    out = {k: [] for k in pool}
    if share is None:
        everything = sorted((k, c) for k in pool for c in pool[k])
        for k, c in sorted(rng.sample(everything, budget)):
            out[k].append(c)
        return out, "pooled"
    for k in sorted(pool):
        calls = sorted(pool[k])
        n = share[k]
        out[k] = calls if n >= len(calls) else sorted(rng.sample(calls, n))
    return out, "allocated"


# ------------------------------------------------------------ the manifest

def key(run) -> str:
    return "/".join(run)


def done(rec) -> bool:
    """A manifest record that stands: a run with an exit code (or a
    timeout), a version obtained; not a skipped run, not a FAILED
    version, which the next pass repeats without --redo."""
    if not rec:
        return False
    if "skipped" in rec or rec.get("how") == "FAILED":
        return False
    return True


def read_manifest(root: pathlib.Path) -> dict:
    path = root / "manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created": now(), "trees": {}, "old": {}, "new": {}, "runs": {}}


def write_manifest(root: pathlib.Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest["written"] = now()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def interpreter_of(tool: list[str], python: str) -> str:
    """The interpreter the tool runs on: the `python` beside a venv's
    script (found on PATH when the tool is a bare name), the tool itself
    when it is an interpreter (`python -m fixer`), else `python`."""
    found = tool[0] if pathlib.Path(tool[0]).exists() \
        else shutil.which(tool[0])
    if found is None:
        return python
    beside = pathlib.Path(found).parent / "python"
    return str(beside) if beside.is_file() else found


def tool_record(tool: list[str], python: str) -> dict:
    """The tool's version, its griffe and libcst, the interpreter it
    runs on, the repository's commit and whether the tree was clean
    (None when git could not say)."""
    rec = {"tool": tool, "python_for_versions": python}
    repo = pathlib.Path(__file__).resolve().parents[1]

    def run(cmd, cwd=None):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=120, cwd=cwd)
            return (r.stdout or r.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"not run: {e}"
    rec["version"] = run(tool + ["--version"])
    interpreter = interpreter_of(tool, python)
    rec["interpreter"] = interpreter
    rec["interpreter_version"] = run([interpreter, "--version"])
    rec["libraries"] = run([interpreter, "-c",
                            "import importlib.metadata as m; print('griffe', "
                            "m.version('griffe'), 'libcst', "
                            "m.version('libcst'), "
                            "'packaging', m.version('packaging'))"])
    rec["commit"] = run(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    status = run(["git", "status", "--porcelain"], cwd=repo)
    rec["tree_clean"] = None if status.startswith("not run") else status == ""
    return rec


# ------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stages", nargs="+",
                    choices=["plan", "resolve", "fetch", "old", "new", "run",
                             "count", "draw"])
    ap.add_argument("--sample", type=pathlib.Path, default=SAMPLE)
    ap.add_argument("--bundle", type=pathlib.Path, default=BUNDLE)
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter whose pip installs the old and "
                         "new versions (default: this one)")
    ap.add_argument("--tool", default="breakrank-fix",
                    help="the command that runs the fixer (default: "
                         "breakrank-fix; 'python -m fixer' also works)")
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    ap.add_argument("--redo", action="store_true",
                    help="repeat what is already done")
    ap.add_argument("--only", help="restrict run to one sample package")
    args = ap.parse_args(argv)

    try:
        runs, olds, news = plan(args.sample, args.bundle)
    except PlanError as e:
        print(e, file=sys.stderr)
        return 2
    root = args.root
    manifest = read_manifest(root)
    tool = args.tool.split()
    samples = sorted({r[0] for r in runs})

    for stage in args.stages:
        print(f"\n== {stage}")
        if stage == "plan":
            print(f"{len(runs)} run(s) over {len(samples)} sample "
                  f"package(s); {len(olds)} old version(s), {len(news)} new "
                  "version(s)")
            per = collections.Counter(r[0] for r in runs)
            for s in samples:
                print(f"  {s}: {per[s]} run(s)")
            manifest["plan"] = {"runs": len(runs), "olds": len(olds),
                                "news": len(news), "sample": str(args.sample)}
            write_manifest(root, manifest)

        elif stage == "resolve":
            for s in samples:
                if s in manifest["trees"] and not args.redo:
                    continue
                info = pypi(PYPI.format(s))
                version, f, kind = choose_release(info)
                rec = {"version": version, "kind": kind, "resolved": now()}
                if f is not None:
                    rec.update(filename=f["filename"], url=f["url"],
                               size=f["size"], sha256=f["digests"]["sha256"],
                               upload_time=f.get("upload_time_iso_8601"))
                manifest["trees"][s] = rec
                print(f"  {s}: {version} {kind or 'NO SOURCE'} "
                      f"{rec.get('filename', '')}")
            write_manifest(root, manifest)

        elif stage == "fetch":
            for s in samples:
                rec = manifest["trees"].get(s)
                if rec is None or rec.get("kind") is None:
                    print(f"  {s}: nothing to fetch (resolve first, or no "
                          "source)")
                    continue
                if rec.get("tree") and not args.redo:
                    continue
                archive = root / "trees" / s / rec["filename"]
                try:
                    size = download(rec["url"], archive, rec["sha256"])
                    tree = unpack(archive, root / "trees" / s / "src")
                except (RuntimeError, OSError, tarfile.TarError,
                        zipfile.BadZipFile) as e:
                    rec.update(fetch_error=f"{type(e).__name__}: {e}")
                    print(f"  {s}: FAILED {rec['fetch_error']}")
                    continue
                rec.update(fetched=now(), bytes=size, tree=str(tree))
                print(f"  {s}: {size:,} bytes -> {tree}")
            write_manifest(root, manifest)

        elif stage in ("old", "new"):
            pairs = olds if stage == "old" else news
            scratch = root / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            for pkg, version in pairs:
                k = f"{pkg}-{version}"
                dest = root / stage / k
                if done(manifest[stage].get(k)) and not args.redo \
                        and dest.is_dir():
                    continue
                if dest.exists():
                    shutil.rmtree(dest)
                try:
                    rec = obtain(args.python, pkg, version, dest, scratch)
                except (RuntimeError, OSError, tarfile.TarError,
                        zipfile.BadZipFile) as e:
                    rec = {"how": "FAILED",
                           "error": f"{type(e).__name__}: {e}"}
                rec["obtained"] = now()
                manifest[stage][k] = rec
                print(f"  {k}: {rec['how']}"
                      + (f" ({rec.get('error') or rec.get('sdist', '')})"
                         if rec["how"] != "wheel" else ""))
                write_manifest(root, manifest)

        elif stage == "run":
            manifest["tool"] = tool_record(tool, args.python)
            for run in runs:
                sample, pkg, frm, to = run
                if args.only and sample != args.only:
                    continue
                if done(manifest["runs"].get(key(run))) and not args.redo:
                    continue
                tree = manifest["trees"].get(sample, {}).get("tree")
                old = manifest["old"].get(f"{pkg}-{frm}", {})
                if not tree or old.get("how") in (None, "FAILED"):
                    manifest["runs"][key(run)] = {
                        "exit": None, "skipped": "no tree" if not tree
                        else "no old version"}
                    print(f"  {key(run)}: skipped, "
                          f"{manifest['runs'][key(run)]['skipped']}")
                    continue
                previous = manifest["runs"].get(key(run))
                if previous and previous.get("commit"):
                    stem = root / sample / f"{pkg}-{frm}-{to}"
                    for ext in (".txt", ".diff"):
                        old_path = stem.with_suffix(ext)
                        if old_path.is_file():
                            old_path.rename(
                                stem.parent
                                / f"{stem.name}.{previous['commit']}{ext}")
                rec = run_one(tool, run, pathlib.Path(tree),
                              root / "old" / f"{pkg}-{frm}", root,
                              timeout=args.timeout)
                rec["ran"] = now()
                rec["tool_version"] = manifest["tool"].get("version")
                rec["commit"] = manifest["tool"].get("commit")
                if previous and previous.get("commit"):
                    rec["superseded"] = previous["commit"]
                manifest["runs"][key(run)] = rec
                print(f"  {key(run)}: exit {rec['exit']} in "
                      f"{rec['seconds']} s" + (" TIMED OUT" if rec["timed_out"]
                                               else ""))
                write_manifest(root, manifest)

        elif stage == "count":
            rows = count(root, runs)
            print("\n".join(totals(rows)))
            print(f"{root / 'counts.csv'}: {len(rows)} row(s)")

        elif stage == "draw":
            pools = {"rewritten": {}, "withheld": {}}
            for run in runs:
                path = root / run[0] / f"{run[1]}-{run[2]}-{run[3]}.txt"
                if not path.is_file():
                    continue
                ds = decisions(path.read_text(encoding="utf-8"))
                for kind in pools:
                    pools[kind][key(run)] = [
                        (d["file"], d["line"], d["column"], d["use"],
                         d["call"], d["symbol"]) for d in ds
                        if d["kind"] == kind]
            drawn_rw, how_rw = draw(pools["rewritten"], REWRITTEN,
                                    SEED_REWRITTEN)
            drawn_wh, how_wh = draw(pools["withheld"], WITHHELD, SEED_WITHHELD)
            with (root / "draw.csv").open("w", encoding="utf-8",
                                          newline="") as f:
                w = csv.writer(f)
                w.writerow(["kind", "sample", "changed", "from", "to",
                            "file", "line", "column", "use", "symbol", "call"])
                for kind, drawn in (("rewritten", drawn_rw),
                                    ("withheld", drawn_wh)):
                    for k in sorted(drawn):
                        for file, line, col, use, call, symbol in drawn[k]:
                            w.writerow([kind, *k.split("/"), file, line, col,
                                        use, symbol, call])
            manifest["draw"] = {"rewritten": REWRITTEN, "withheld": WITHHELD,
                                "seeds": [SEED_REWRITTEN, SEED_WITHHELD],
                                "how": {"rewritten": how_rw,
                                        "withheld": how_wh},
                                "drawn": now()}
            for kind, drawn, pool, how in (
                    ("rewritten", drawn_rw, pools["rewritten"], how_rw),
                    ("withheld", drawn_wh, pools["withheld"], how_wh)):
                total = sum(len(v) for v in pool.values())
                got = sum(len(v) for v in drawn.values())
                print(f"  {kind}: {got} drawn of {total} in "
                      f"{sum(1 for v in pool.values() if v)} run(s) with any, "
                      f"{how}")
            write_manifest(root, manifest)
            print(f"{root / 'draw.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
