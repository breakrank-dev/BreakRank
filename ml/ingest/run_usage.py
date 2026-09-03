"""
Track B at scale — scan the top N packages and build the usage index.

    python ml/ingest/run_usage.py --packages 1500
    python ml/ingest/run_usage.py --packages 1500 --restart

Reads  data/changes.csv        (Track A — defines which module roots we track)
Writes data/usage_pairs.csv    one row per (scanning package, symbol it uses)
       data/usage.csv          one row per symbol: how many DISTINCT packages use it
       data/usage_done.txt     resume file
       data/usage_failures.csv what went wrong, with reasons

Design notes, learned the hard way on Day 2:

* RESUMABLE and PARALLEL, same as run_ingest. Only the parent writes files.

* PER-PACKAGE TIMEOUT. In the Track A run, cython alone burned 26 minutes
  and eight packages ate over half the wall clock. Here every worker gets a
  hard alarm (default 120s); a package that blows it is logged as a failure
  and the run moves on. Bounding the unit of work is what makes the nightly
  job's runtime predictable.

* SELF-USE IS EXCLUDED. pandas's own source says `import pandas` hundreds of
  times. Count that and every package "uses" its own symbols, every symbol of
  a scanned package gets labelled important, and the labels are quietly
  wrecked. A package is not downstream of itself.

* We count DISTINCT PACKAGES per symbol, never call sites. One package
  calling requests.get a thousand times is still ONE package that breaks.
  Per-call-site counts make popular symbols look absurdly important, and
  nothing errors to tell you so.

* Latest version only. The question is "who depends on this symbol TODAY?",
  so history is irrelevant here — that is Track A's job.
"""

import argparse
import concurrent.futures as cf
import os
import pathlib
import shutil
import signal
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.ingest.api_extract import find_import_names          # noqa: E402
from ml.ingest.download import download_and_extract, list_releases  # noqa: E402
from ml.ingest.packages import get_top_packages              # noqa: E402
from ml.ingest.usage_index import scan_package               # noqa: E402

DATA = pathlib.Path("data")
CHANGES = DATA / "changes.csv"
PAIRS = DATA / "usage_pairs.csv"
USAGE = DATA / "usage.csv"
DONE = DATA / "usage_done.txt"
FAILURES = DATA / "usage_failures.csv"
SCAN_DIR = DATA / "usage_sdists"

FAILURE_COLS = ["package", "stage", "error_type", "message"]


def tracked_roots() -> set[str]:
    """
    The module roots our candidate changes live in — derived from Track A.

    We never wrote the import names down during ingest, but we do not need
    to: every symbol in changes.csv STARTS with its module name.
    "click.utils.LazyFile" -> "click". Free, and always in sync with the
    data it will be joined against.
    """
    if not CHANGES.exists():
        sys.exit(f"{CHANGES} not found — run ml/ingest/run_ingest.py first. "
                 "Track B labels Track A's rows; without them there is "
                 "nothing to track.")
    symbols = pd.read_csv(CHANGES, usecols=["symbol"])["symbol"]
    return {s.split(".")[0] for s in symbols}


class ScanTimeout(Exception):
    pass


def _alarm(_sig, _frame):
    raise ScanTimeout()


def scan_one(name: str, tracked: set[str], timeout_s: int) -> list[dict]:
    """
    Worker: download one package's latest sdist, scan it, clean up.
    Returns [{"scanner": name, "symbol": ...}, ...]. Raises on failure —
    the parent records why. The alarm turns a pathological package into a
    logged failure instead of a stuck run.
    """
    dest = SCAN_DIR / name
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        releases = list_releases(name, last_n=1)
        if not releases:
            raise RuntimeError("no sdist releases")
        path = download_and_extract(releases[-1]["url"], dest)
        if path is None:
            raise RuntimeError("could not work out the sdist layout")

        # Which modules are this package's OWN? Anything it imports from
        # itself is self-use and must not count.
        own = set(find_import_names(path, name))

        used = scan_package(path, tracked - own)
        return [{"scanner": name, "symbol": s} for s in used
                if s.split(".")[0] not in own]
    finally:
        signal.alarm(0)
        shutil.rmtree(dest, ignore_errors=True)


def aggregate() -> None:
    """usage_pairs -> usage.csv: distinct scanners per symbol."""
    if not PAIRS.exists():
        print("no pairs recorded; nothing to aggregate")
        return
    pairs = pd.read_csv(PAIRS).drop_duplicates()
    usage = (pairs.groupby("symbol")["scanner"].nunique()
                  .rename("user_count").reset_index()
                  .sort_values("user_count", ascending=False))
    usage.to_csv(USAGE, index=False)

    print(f"\n{USAGE}: {len(usage):,} distinct symbols used by someone")
    print("\nMost-used symbols in the ecosystem:")
    for _, r in usage.head(12).iterrows():
        print(f"  {r.user_count:>5}  {r.symbol}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Track B usage index.")
    ap.add_argument("--packages", type=int, default=1500,
                    help="scan the top N PyPI packages")
    ap.add_argument("--timeout", type=int, default=120,
                    help="seconds allowed per package before it is skipped")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    if args.restart:
        for p in (PAIRS, USAGE, DONE, FAILURES):
            p.unlink(missing_ok=True)
        shutil.rmtree(SCAN_DIR, ignore_errors=True)
        print("cleared previous usage run\n")

    tracked = tracked_roots()
    print(f"tracking {len(tracked)} module roots from {CHANGES}")

    done = set()
    if DONE.exists():
        done = {l.strip() for l in DONE.read_text().splitlines() if l.strip()}
    todo = [p for p in get_top_packages(args.packages) if p["name"] not in done]
    workers = max(1, min(args.workers, len(todo) or 1))
    print(f"{args.packages} packages requested, {len(done)} done, "
          f"{len(todo)} to go, {workers} in parallel, "
          f"{args.timeout}s timeout each\n")

    started = time.time()
    n_pairs = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_one, p["name"], tracked, args.timeout): p["name"]
                   for p in todo}
        try:
            for i, fut in enumerate(cf.as_completed(futures), 1):
                name = futures[fut]
                try:
                    rows = fut.result()
                    fails = []
                except ScanTimeout:
                    rows, fails = [], [{"package": name, "stage": "scan",
                                        "error_type": "Timeout",
                                        "message": f"exceeded {args.timeout}s"}]
                except Exception as e:
                    rows, fails = [], [{"package": name, "stage": "scan",
                                        "error_type": type(e).__name__,
                                        "message": " ".join(str(e).split())[:200]}]
                    if type(e).__name__ == "UnexpectedError":
                        traceback.print_exc()

                if rows:
                    pd.DataFrame(rows).to_csv(PAIRS, mode="a",
                                              header=not PAIRS.exists(), index=False)
                if fails:
                    pd.DataFrame(fails)[FAILURE_COLS].to_csv(
                        FAILURES, mode="a", header=not FAILURES.exists(), index=False)
                with DONE.open("a") as f:
                    f.write(name + "\n")

                n_pairs += len(rows)
                rate = i / max(time.time() - started, 1e-9)
                eta = (len(todo) - i) / rate / 60
                flag = "" if not fails else f"  ({fails[0]['error_type']})"
                print(f"[{i:>4}/{len(todo)}] {name:<30} {len(rows):>5} symbols"
                      f"   eta {eta:>5.1f} min{flag}", flush=True)
        except KeyboardInterrupt:
            print("\nStopping — progress is saved. Rerun to resume.")
            pool.shutdown(wait=False, cancel_futures=True)

    print(f"\n{n_pairs:,} usage pairs recorded in "
          f"{(time.time() - started) / 60:.1f} minutes")
    aggregate()


if __name__ == "__main__":
    main()
