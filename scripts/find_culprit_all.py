"""
Find EVERY package that trips macOS XProtect — in a single run.

    python scripts/find_culprit_all.py

find_culprit.py dies at the FIRST culprit. This one doesn't. It runs each
package inside its own isolated child process (its own session), so when
XProtect kills that child, THIS script survives and moves on. One run
catalogs every flagged package instead of stopping at the first.

It only checks the packages you have NOT already ingested (top-500 minus
what's in changes.csv), so it goes straight to the unknown ones.

You will see the popup a few times — that is the point, it means the
finder caught one. Click Done each time; it keeps running. When it
finishes, the full list is in data/culprits.txt. Send me that file.

Nothing here executes package code (same allow_inspection=False as the
pipeline); it downloads, extracts and statically reads, exactly what the
real run does.
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CULPRITS = DATA / "culprits.txt"

# The child: download + extract + read ONE package, then clean up. Run as
# an isolated process so a kill cannot reach the parent. Extracts the last
# 3 releases (enough to surface a flagged file, which is stable across
# versions) and statically loads the newest.
WORKER = r'''
import sys, pathlib, shutil
sys.path.insert(0, %r)
from ml.ingest.download import list_releases, download_and_extract
from ml.ingest.api_extract import resolve_layout, load_all_modules
name = sys.argv[1]
tmp = pathlib.Path(%r) / "culprit_tmp" / name
try:
    rels = list_releases(name, last_n=3)
    newest = None
    for r in rels:
        d = download_and_extract(r["url"], tmp / r["version"])
        if d:
            newest = d
    if newest is not None:
        here, mods = resolve_layout(newest, name)
        load_all_modules(here, mods)
    print("OK", flush=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
'''  % (str(ROOT), str(DATA))


def missing_packages() -> list[str]:
    import pandas as pd
    top = [r["project"]
           for r in json.loads((DATA / "top-pypi-packages.json").read_text())
           ["rows"][:500]]
    done = set()
    if (DATA / "changes.csv").exists():
        done = set(pd.read_csv(DATA / "changes.csv", usecols=["package"])
                   ["package"].unique())
    return [p for p in top if p not in done]


def main() -> None:
    pkgs = missing_packages()
    print(f"Checking {len(pkgs)} not-yet-done packages, one isolated at a "
          f"time.\nPopups may appear — click Done, it keeps going. Culprits "
          f"land in\n{CULPRITS} as they're found.\n")

    culprits: list[str] = []
    for i, name in enumerate(pkgs, 1):
        try:
            p = subprocess.run(
                [sys.executable, "-c", WORKER, name],
                start_new_session=True,      # isolate: a kill can't reach us
                capture_output=True, text=True, timeout=1800)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            print(f"[{i:>3}/{len(pkgs)}] {name:<34} timeout", flush=True)
            continue

        # A negative return code means the child was terminated by a signal.
        # XProtect kills with SIGKILL (-9). A normal failure (bad sdist, no
        # module) exits positive and is not a culprit.
        if rc < 0:
            culprits.append(name)
            CULPRITS.write_text("\n".join(culprits) + "\n")
            print(f"[{i:>3}/{len(pkgs)}] {name:<34} *** XPROTECT KILLED "
                  f"(signal {-rc})", flush=True)
        elif "OK" in p.stdout:
            print(f"[{i:>3}/{len(pkgs)}] {name:<34} ok", flush=True)
        else:
            print(f"[{i:>3}/{len(pkgs)}] {name:<34} (benign error)", flush=True)

    print("\n" + "=" * 60)
    if culprits:
        print(f"{len(culprits)} package(s) trip XProtect on this Mac:")
        for c in culprits:
            print(f"   {c}")
        print(f"\nSaved -> {CULPRITS}. Send me this file.")
    else:
        print("No package tripped XProtect. Nothing to skip.")


if __name__ == "__main__":
    main()
