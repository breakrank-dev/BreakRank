"""
The real ingest, made immune to macOS XProtect.

    python scripts/ingest_resilient.py
    python scripts/ingest_resilient.py --workers 1     (strictly one at a time)

run_ingest.py dies the moment XProtect kills one worker, because its
workers share a session with the parent and the kill takes everything
down. This runs every package in its OWN isolated child process (its own
session). When XProtect kills a child, THIS parent survives, writes the
package name to data/xprotect_killed.txt, and carries on. Every clean
package produces real rows, exactly like run_ingest — same columns, same
files — so the output is a drop-in changes.csv.

At the end you have:
    data/changes.csv           every package XProtect left alone
    data/xprotect_killed.txt   the ones it killed  ->  send me this file

I process the killed ones on Linux (no XProtect) and we merge. Nothing
here runs package code — same allow_inspection=False as the pipeline.

Resumes from data/done.txt like run_ingest, so it picks up where any
earlier run stopped. Do NOT use --restart with this; it continues.
"""

import argparse
import concurrent.futures as cf
import json
import pathlib
import shutil
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.ingest.run_ingest import (append_rows, load_done, mark_done,    # noqa: E402
                                  CHANGE_COLS, FAILURE_COLS, PACKAGE_COLS,
                                  RELEASE_COLS, CHANGES, FAILURES, PACKAGES,
                                  RELEASES, SDISTS)
from ml.ingest.packages import get_top_packages                         # noqa: E402

DATA = ROOT / "data"
KILLED = DATA / "xprotect_killed.txt"
TMP = DATA / "resilient_tmp"

# The child. Runs the pipeline's own process_package for ONE package and
# dumps the four result lists to a JSON file. Isolated in its own session
# so an XProtect kill stops at this process.
WORKER = r'''
import sys, json
sys.path.insert(0, %r)
from ml.ingest.run_ingest import process_package
name, rank, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows, failures, meta, rels = process_package(name, rank, 10)
json.dump({"rows": rows, "failures": failures, "meta": meta, "rels": rels},
          open(out, "w"))
print("OK", flush=True)
''' % (str(ROOT),)

_lock = threading.Lock()


def run_one(name: str, rank: int, timeout: int) -> tuple[str, str, dict | None]:
    """Run one package isolated. Returns (name, status, payload)."""
    out = TMP / f"{name}.json"
    try:
        p = subprocess.run(
            [sys.executable, "-c", WORKER, name, str(rank), str(out)],
            start_new_session=True,       # a kill cannot reach the parent
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        shutil.rmtree(SDISTS / name, ignore_errors=True)
        return name, "timeout", None

    if p.returncode < 0:                  # terminated by a signal = XProtect
        shutil.rmtree(SDISTS / name, ignore_errors=True)  # child never cleaned
        return name, "killed", None
    if p.returncode == 0 and out.exists():
        try:
            payload = json.load(open(out))
        finally:
            out.unlink(missing_ok=True)
        return name, "ok", payload
    return name, "error", {"stderr": p.stderr[-300:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", type=int, default=500)
    ap.add_argument("--workers", type=int, default=3,
                    help="isolated children at once. Each is its own "
                         "session, so a kill only ever costs one package.")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    TMP.mkdir(parents=True, exist_ok=True)
    packages = get_top_packages(args.packages)
    done = load_done()
    todo = [p for p in packages if p["name"] not in done]
    killed_so_far = ([l.strip() for l in KILLED.read_text().splitlines()
                      if l.strip()] if KILLED.exists() else [])

    print(f"{len(packages)} requested, {len(done)} already done, "
          f"{len(todo)} to go, {args.workers} isolated at a time.")
    print("XProtect popups are expected — click Done, this keeps running.\n")

    started = time.time()
    n = 0
    total_rows = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, p["name"], p["rank"], args.timeout): p
                for p in todo}
        for fut in cf.as_completed(futs):
            name, status, payload = fut.result()
            n += 1
            with _lock:
                if status == "ok":
                    append_rows(CHANGES, payload["rows"], CHANGE_COLS)
                    append_rows(RELEASES, payload["rels"], RELEASE_COLS)
                    append_rows(FAILURES, payload["failures"], FAILURE_COLS)
                    append_rows(PACKAGES, payload["meta"], PACKAGE_COLS)
                    total_rows += len(payload["rows"])
                    tag = f"{len(payload['rows']):>5} rows"
                elif status == "killed":
                    killed_so_far.append(name)
                    KILLED.write_text("\n".join(killed_so_far) + "\n")
                    append_rows(FAILURES, [{
                        "package": name, "stage": "skipped", "detail": "",
                        "error_type": "XProtectBlocked",
                        "message": "killed by macOS XProtect; "
                                   "backfilled from Linux"}], FAILURE_COLS)
                    tag = "*** XPROTECT KILLED -> saved to xprotect_killed.txt"
                elif status == "timeout":
                    append_rows(FAILURES, [{
                        "package": name, "stage": "timeout", "detail": "",
                        "error_type": "Timeout",
                        "message": f"exceeded {args.timeout}s"}], FAILURE_COLS)
                    tag = "timeout"
                else:
                    append_rows(FAILURES, [{
                        "package": name, "stage": "pipeline", "detail": "",
                        "error_type": "UnexpectedError",
                        "message": (payload or {}).get("stderr", "")[:200]}],
                        FAILURE_COLS)
                    tag = "error"
                mark_done(name)

            rate = n / max(time.time() - started, 1e-9)
            eta = (len(todo) - n) / rate / 60
            print(f"[{n:>4}/{len(todo)}] {name:<30} {tag}   eta {eta:>5.1f} min",
                  flush=True)

    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 62)
    print(f"  {total_rows:,} new rows.  Killed by XProtect: {len(killed_so_far)}")
    if killed_so_far:
        print("  " + ", ".join(killed_so_far))
        print(f"\n  Send me {KILLED} and I'll do those on Linux.")
    else:
        print("  None killed — the dataset is complete on this machine.")


if __name__ == "__main__":
    main()
