"""
Find which package trips macOS XProtect during the ingest.

    python scripts/find_culprit.py

The full run dies silently around package 233 because macOS blocks
something it flags as malware and kills the process. This narrows down
WHAT. It does the exact same thing the pipeline does — download, extract,
static-load with griffe — but for ONE package at a time, in rank order,
and it writes the package name to disk BEFORE touching it.

So when macOS kills this process, the answer is already saved:

    cat data/culprit_current.txt

holds the last package it tried — the one with the flagged file. Because
nothing runs in parallel here, there is no ambiguity: that package is it.

TWO POSSIBLE OUTCOMES, both useful:

  * It dies on a package  -> that package carries a file macOS flags.
    We add it to a skip list and the real run never touches it.

  * It runs all the way to the end with no block -> it is NOT one bad
    file, it is the *pattern* of the full run (many parallel downloads at
    once). Then the fix is fewer workers, not a skip. Either way we learn
    the real cause instead of guessing.

Nothing here executes package code — same allow_inspection=False as the
pipeline. It only reads.
"""

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.ingest.packages import get_top_packages          # noqa: E402
from ml.ingest.download import (list_releases,            # noqa: E402
                                download_and_extract)
from ml.ingest.api_extract import (resolve_layout,        # noqa: E402
                                   load_all_modules)

DATA = pathlib.Path("data")
CUR = DATA / "culprit_current.txt"
TMP = DATA / "culprit_sdists"

# The band where every death clustered. Wide enough to be safe; it stops
# the instant macOS blocks, so covering extra packages costs nothing.
LO, HI = 215, 290


def main() -> None:
    DATA.mkdir(exist_ok=True)
    packages = get_top_packages(HI)
    band = packages[LO - 1:HI]
    print(f"Checking ranks {LO}-{HI}, one at a time. Watch for the macOS "
          f"popup.\nWhen it appears, the culprit is already saved in "
          f"{CUR}.\n")

    for p in band:
        name, rank = p["name"], p["rank"]
        # Written BEFORE we touch the package, and flushed to disk, so a
        # kill cannot erase it.
        CUR.write_text(f"{rank} {name}\n")
        print(f"[{rank:>3}] {name:<34} ", end="", flush=True)
        try:
            rels = list_releases(name, last_n=10)
            n = 0
            for r in rels:
                d = download_and_extract(r["url"], TMP / name / r["version"])
                if d:
                    here, mods = resolve_layout(d, name)
                    load_all_modules(here, mods)   # static read only
                    n += 1
            shutil.rmtree(TMP / name, ignore_errors=True)
            print(f"ok ({n} versions read)", flush=True)
        except Exception as e:
            print(f"skipped: {type(e).__name__}", flush=True)
            shutil.rmtree(TMP / name, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Reached the end with NO block.")
    print("That means it is not one bad file — it is the parallel-download")
    print("pattern of the full run. The fix is fewer workers, not a skip.")


if __name__ == "__main__":
    main()
