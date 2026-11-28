"""
Where the rows the deletion step marks come from: today's files, or loads
before them.

    python scripts/db_drop_split.py

READ ONLY. Every statement in this file is a SELECT, so it is safe to run
against the live database before or after scripts/db_drop_versions.py.

VARAD'S QUESTION (15 Oct, NOTES §33.6). The deletion step's dry run marked
2,336 rows on Neon where §33.2 had counted 1,815 in labelled.csv, and
1,120 releases left with nothing where §33.2 had counted 895. Both
differences are the rows earlier loads left behind: the loader upserts and
never removes (§8, §13), so the table is the union of every run ever
made, and a release that aged out of the six-release window keeps its
rows, its version string among them. This splits both counts by the
current data/changes.csv:

  - a marked row whose (package, version, symbol, kind) is a row of
    changes.csv is today's. Marked rows are attribute changes, which have
    no sub_target, so the key needs none;
  - the rest came from earlier loads: either in a release today's files
    still name (changes.csv has the release, not this row) or in one that
    left the window;
  - a release left with nothing is today's if changes.csv names it as a
    version_to; the same releases are what §33.2's all-clear counted on
    the file, and the two sets are compared both ways, with the odd ones
    named.

The numbers are read from the table as it is now, so run this BEFORE the
deletion to split the rows; after it the marked rows are gone and only the
file's side of the split can be counted.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import connect, load_frames, not_a_change  # noqa: E402
from scripts.db_drop_versions import fetch, marked, plan  # noqa: E402

SHOW = 10


def split(rows: pd.DataFrame, changes: pd.DataFrame) -> dict:
    """The marked rows and the emptied releases of a table (as fetch()
    reads it), split by today's changes.csv. Pure, so the test can check
    it; main() prints it."""
    m = marked(rows)
    go = m[(m["rule"] != "") & ~m["fixture"]]
    chg = changes.astype({"package": str, "version_to": str, "symbol": str,
                          "kind": str})
    today_keys = set(zip(chg["package"], chg["version_to"], chg["symbol"],
                         chg["kind"]))
    live = set(zip(chg["package"], chg["version_to"]))
    go_key = list(zip(go["package"].astype(str), go["version"].astype(str),
                      go["symbol"].astype(str), go["kind"].astype(str)))
    # Boolean arrays, not lists: a frame indexed by an empty LIST selects
    # no columns, where an empty boolean array selects no rows.
    today = np.array([k in today_keys for k in go_key], dtype=bool)
    earlier = go[~today]
    earlier_live = earlier[np.array([(p, v) in live for p, v in
                                     zip(earlier["package"].astype(str),
                                         earlier["version"].astype(str))],
                                    dtype=bool)]

    # The file's own count of the same rows, as §33.2 made it, and once
    # per key as the table holds them.
    file_marked = not_a_change(chg)
    file_keys = set(k for k, t in zip(zip(chg["package"], chg["version_to"],
                                           chg["symbol"], chg["kind"]),
                                       file_marked) if t)
    # The file's all-clear: releases every row of which is marked.
    per_release = pd.DataFrame({"rel": list(live_of(chg)),
                                "gone": list(file_marked)})
    all_rows = per_release.groupby("rel")["gone"].all()
    file_clear = set(all_rows[all_rows].index)

    p = plan(rows)
    rel_of = (m.drop_duplicates("release_id").set_index("release_id")
              [["package", "version"]].astype(str))
    emptied = {(rel_of.at[r, "package"], rel_of.at[r, "version"])
               for r in p["emptied"] if r in rel_of.index}
    emptied_today = {r for r in emptied if r in live}
    return {
        "marked": int(len(go)),
        "today": int(today.sum()),
        "earlier": int(len(earlier)),
        "earlier_live": int(len(earlier_live)),
        "earlier_aged": int(len(earlier) - len(earlier_live)),
        "file_marked": int(file_marked.sum()),
        "file_keys": len(file_keys),
        "emptied": len(emptied),
        "emptied_today": len(emptied_today),
        "emptied_earlier": len(emptied) - len(emptied_today),
        "file_clear": len(file_clear),
        "both": len(file_clear & emptied_today),
        "file_only": sorted(file_clear - emptied_today),
        "table_only": sorted(emptied_today - file_clear),
    }


def live_of(chg: pd.DataFrame):
    return zip(chg["package"], chg["version_to"])


def describe(s: dict) -> str:
    lines = [
        f"\nrows the deletion step marks (not breakrank-fixture's): "
        f"{s['marked']:,}",
        f"  rows of today's changes.csv:        {s['today']:>7,}   (the file "
        f"itself: {s['file_marked']:,} by rule, {s['file_keys']:,} distinct "
        "keys)",
        f"  from earlier loads:                 {s['earlier']:>7,}",
        f"    in releases today's files name:   {s['earlier_live']:>7,}   "
        "(rows this run did not produce)",
        f"    in releases outside today's files:{s['earlier_aged']:>7,}   "
        "(aged out of the window)",
        f"\nreleases left with nothing once they go: {s['emptied']:,}",
        f"  named by today's files:             {s['emptied_today']:>7,}",
        f"  earlier loads only:                 {s['emptied_earlier']:>7,}",
        f"\n  the file's own all-clear, releases every row of which is "
        f"marked: {s['file_clear']:,}",
        f"  in both:  {s['both']:,}   file only: {len(s['file_only']):,}   "
        f"table only: {len(s['table_only']):,}",
    ]
    for name, odd in (("file only", s["file_only"]),
                      ("table only", s["table_only"])):
        if odd:
            lines.append(f"  {name}: " + ", ".join(
                f"{p} {v}" for p, v in odd[:SHOW])
                + (f" ... and {len(odd) - SHOW:,} more"
                   if len(odd) > SHOW else ""))
    if s["file_only"]:
        lines.append("  (a release in the file's all-clear but not left "
                     "empty in the table has other rows\n  there from an "
                     "earlier load; db_prune's dry run classifies those)")
    return "\n".join(lines)


def main() -> None:
    changes, _usage, _packages = load_frames()
    with connect().connect() as conn:
        rows, _has_status = fetch(conn)
    print(describe(split(rows, changes)))
    print("\nRead only; nothing was changed.")


if __name__ == "__main__":
    main()
