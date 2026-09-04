"""
The real ingest pipeline — Track A at scale.

day1_demo.py proved the idea on 6 packages. This is the version that survives
300 packages: it resumes after a crash, cleans up after itself so your disk
does not fill, and records every failure with a reason instead of losing it.

    python ml/ingest/run_ingest.py --packages 50
    python ml/ingest/run_ingest.py --packages 300 --workers 8
    python ml/ingest/run_ingest.py --packages 300 --restart

Outputs, all under data/ (which is gitignored):

    changes.csv           one row per candidate breaking change
    failures.csv          one row per thing that went wrong, with a reason
    done.txt              packages already processed — this is the resume file

Three design decisions worth understanding, because an interviewer will ask
why a data pipeline looks like this:

1. RESUMABLE. 300 packages is over an hour. Laptops sleep, wifi drops,
   Ctrl-C happens. Every finished package is appended to done.txt, and a
   rerun skips those. You never redo an hour of work.

2. CLEANS UP. Six packages cost 210 MB of extracted source. Three hundred
   would be several GB. Each package's source is deleted the moment its
   diffs are written, so peak disk stays at roughly one package.

3. PARALLEL. Packages do not depend on each other, and each one is part
   waiting on PyPI and part burning CPU in griffe. Separate processes
   (not threads — griffe parsing holds the GIL) cut a 300-package run
   from about an hour to well under twenty minutes. Workers return rows;
   only the parent writes files, so the CSVs can never interleave.

4. LOGS FAILURES AS DATA. Not printed and forgotten — written to a CSV with
   a stage and an error type, so you can group them and find the single fix
   that recovers the most packages. That table is week 2's real work.
"""

import argparse
import concurrent.futures as cf
import os
import pathlib
import shutil
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.ingest.api_extract import diff_versions, find_import_names  # noqa: E402
from ml.ingest.download import download_and_extract, list_releases  # noqa: E402
from ml.ingest.packages import get_top_packages          # noqa: E402

DATA = pathlib.Path("data")
CHANGES = DATA / "changes.csv"
FAILURES = DATA / "failures.csv"
DONE = DATA / "done.txt"
SDISTS = DATA / "sdists"

CHANGE_COLS = [
    "package", "package_rank", "version_from", "version_to", "symbol", "kind",
    "is_private", "in_dunder_all", "module_depth", "is_top_level",
    "name_length", "released_at", "explanation",
]
FAILURE_COLS = ["package", "stage", "detail", "error_type", "message"]


# --------------------------------------------------------------- small helpers

def append_rows(path: pathlib.Path, rows: list[dict], columns: list[str]) -> None:
    """Append to a CSV, writing the header only if the file is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[columns]
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_done() -> set[str]:
    if not DONE.exists():
        return set()
    return {line.strip() for line in DONE.read_text().splitlines() if line.strip()}


def mark_done(package: str) -> None:
    DONE.parent.mkdir(parents=True, exist_ok=True)
    with DONE.open("a") as f:
        f.write(package + "\n")


def short(e: Exception, limit: int = 200) -> str:
    """One-line error message, no newlines, safe for a CSV cell."""
    return " ".join(str(e).split())[:limit]


# ------------------------------------------------------------------ one package

def process_package(name: str, rank: int, n_versions: int) -> tuple[list[dict], list[dict]]:
    """
    Diff every consecutive version pair of one package.

    Returns (rows, failures) and writes nothing. The caller does all the
    file I/O, which is what lets several of these run in parallel processes
    without two workers appending to the same CSV at the same time.

    Never raises — a package that explodes is recorded and the run
    continues. That is the whole point.
    """
    failures: list[dict] = []
    base = SDISTS / name

    # --- stage 1: what versions exist? -------------------------------------
    try:
        releases = list_releases(name, last_n=n_versions)
    except Exception as e:
        failures.append({"package": name, "stage": "list_releases", "detail": "",
                         "error_type": type(e).__name__, "message": short(e)})
        return [], failures

    if len(releases) < 2:
        failures.append({"package": name, "stage": "list_releases",
                         "detail": f"{len(releases)} sdist releases",
                         "error_type": "TooFewReleases",
                         "message": "needs at least 2 versions with a source distribution"})
        return [], failures

    # --- stage 2: download every version once ------------------------------
    paths: dict[str, pathlib.Path] = {}
    for rel in releases:
        try:
            p = download_and_extract(rel["url"], base / rel["version"])
            if p is None:
                raise RuntimeError("could not work out the sdist layout")
            paths[rel["version"]] = p
        except Exception as e:
            failures.append({"package": name, "stage": "download",
                             "detail": rel["version"],
                             "error_type": type(e).__name__, "message": short(e)})

    if not paths:
        return [], failures

    # --- stage 3: what is this thing actually called when you import it? ---
    # The PyPI name and the import name disagree for ~30% of packages
    # (typing-extensions -> typing_extensions, pyyaml -> yaml). Resolve it
    # once, from the newest version we managed to download.
    newest = max(paths, key=lambda v: [r["version"] for r in releases].index(v))
    modules = find_import_names(paths[newest], name)
    if not modules:
        failures.append({"package": name, "stage": "resolve_module",
                         "detail": newest, "error_type": "NoPythonModule",
                         "message": "no importable module in the sdist — "
                                    "compiled-only or not a Python package"})
        shutil.rmtree(base, ignore_errors=True)
        return [], failures
    # --- stage 4: diff consecutive pairs -----------------------------------
    rows: list[dict] = []
    for old, new in zip(releases, releases[1:]):
        vf, vt = old["version"], new["version"]
        if vf not in paths or vt not in paths:
            continue                      # already logged as a download failure
        try:
            found = diff_versions(name, paths[vf], paths[vt])
        except Exception as e:
            failures.append({"package": name, "stage": "griffe",
                             "detail": f"{vf} -> {vt}",
                             "error_type": type(e).__name__, "message": short(e)})
            continue

        for r in found:
            r["package_rank"] = rank
            r["version_from"] = vf
            r["version_to"] = vt
            r["released_at"] = new["uploaded"][:10]
        rows.extend(found)

    # --- stage 5: clean up -------------------------------------------------
    # Delete the source now, while we still know it is safe to. Skipping this
    # is how a 300-package run fills a laptop and dies at package 180.
    shutil.rmtree(base, ignore_errors=True)

    return rows, failures


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Track A ingest pipeline.")
    ap.add_argument("--packages", type=int, default=50,
                    help="how many of the top PyPI packages to process")
    ap.add_argument("--versions", type=int, default=6,
                    help="how many recent releases per package")
    ap.add_argument("--restart", action="store_true",
                    help="wipe previous output and start from scratch")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="packages to process at once (default: CPU cores - 1)")
    args = ap.parse_args()

    if args.restart:
        for p in (CHANGES, FAILURES, DONE):
            p.unlink(missing_ok=True)
        shutil.rmtree(SDISTS, ignore_errors=True)
        print("cleared previous run\n")

    # Schema guard. The API contract added in_dunder_all (3 Sep 2026), so a
    # changes.csv from before then has a different header. Appending
    # 13-column rows to a 12-column file would not error — pandas would
    # write them and every later read would be quietly misaligned. Refuse
    # loudly instead. Mixed schemas in one CSV are worse than redoing a run.
    if CHANGES.exists():
        with CHANGES.open() as f:
            have = f.readline().strip().split(",")
        if have != CHANGE_COLS:
            sys.exit(
                f"{CHANGES} was written by an older version of this pipeline "
                f"({len(have)} columns; the code now writes {len(CHANGE_COLS)}: "
                "in_dunder_all was added for the API contract).\n"
                "Rerun with --restart to produce one uniform dataset."
            )

    packages = get_top_packages(args.packages)
    done = load_done()
    todo = [p for p in packages if p["name"] not in done]

    workers = max(1, min(args.workers, len(todo) or 1))
    print(f"{len(packages)} packages requested, {len(done)} already done, "
          f"{len(todo)} to go")
    print(f"{args.versions} versions each, {workers} in parallel. "
          f"Output -> {CHANGES}\n")

    started = time.time()
    total_rows = 0
    all_failures: list[dict] = []

    # Packages are independent of each other, and each one is a mix of
    # waiting on PyPI and burning CPU in griffe. Separate processes give
    # real parallelism for both halves — threads would only help the
    # downloads, because griffe parsing holds the GIL.
    #
    # Workers return their rows; only this parent process writes files.
    # Two processes appending to one CSV would interleave and corrupt it.
    done_count = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_package, p["name"], p["rank"], args.versions): p
            for p in todo
        }
        try:
            for fut in cf.as_completed(futures):
                pkg = futures[fut]
                name = pkg["name"]
                try:
                    rows, failures = fut.result()
                except Exception:
                    # A bug in our own code, not in the package. Show it.
                    traceback.print_exc()
                    rows = []
                    failures = [{"package": name, "stage": "pipeline", "detail": "",
                                 "error_type": "UnexpectedError",
                                 "message": "see traceback above"}]

                append_rows(CHANGES, rows, CHANGE_COLS)
                append_rows(FAILURES, failures, FAILURE_COLS)
                all_failures.extend(failures)
                total_rows += len(rows)
                mark_done(name)

                done_count += 1
                flag = "" if not failures else f"  ({len(failures)} failure(s))"
                rate = done_count / max(time.time() - started, 1e-9)
                eta = (len(todo) - done_count) / rate / 60 if rate else 0
                print(f"[{done_count:>4}/{len(todo)}] {name:<26} "
                      f"{len(rows):>5} rows   eta {eta:>5.1f} min{flag}",
                      flush=True)
        except KeyboardInterrupt:
            print("\n\nStopping — finished packages are saved. "
                  "Rerun the same command to pick up where you left off.")
            pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------- summary
    elapsed = time.time() - started
    print(f"\n{'=' * 62}")
    print(f"  {total_rows:,} new rows in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 62}")

    if CHANGES.exists():
        df = pd.read_csv(CHANGES)
        print(f"\n{CHANGES}: {len(df):,} rows total, "
              f"{df['package'].nunique()} packages")
        print("\nTop kinds:")
        for kind, n in df["kind"].value_counts().head(6).items():
            print(f"  {n:>6}  ({n / len(df):>4.0%})  {kind}")

    if FAILURES.exists():
        f = pd.read_csv(FAILURES)
        print(f"\n{FAILURES}: {len(f)} failures across "
              f"{f['package'].nunique()} packages")
        print("\nFailures by stage and error type — THIS is the table to read:")
        grouped = (f.groupby(["stage", "error_type"])
                    .agg(failures=("package", "size"),
                         packages=("package", "nunique"))
                    .sort_values("packages", ascending=False))
        print(grouped.to_string())
        print("\nThe row costing you the most PACKAGES is the one worth fixing first.")

    print(f"\nDisk now: {sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()) / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()
