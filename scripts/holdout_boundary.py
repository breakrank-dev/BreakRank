"""
Where should the holdout start? Counts only: nothing is fitted or scored.

    python scripts/holdout_boundary.py

Reads  data/features.csv and data/holdout.csv (together, every row)
Writes nothing

WHY THIS EXISTS. The first build after the freeze printed that under
`label` the holdout's nDCG@20 would be an average over 8 upgrades, below
the 10-pair gate every measurement in this project has to clear. And F1,
next on the fix list, deletes the version-string rows, which removes
positives and shrinks releases, so every count will fall.

The start date can still move today, because nothing has been scored on
the holdout and no fix on the list has been run. Once either has
happened, moving it would be a choice made with results in view, which is
what the freeze exists to prevent. So it is settled now, by a rule written
down before this script was first run:

    Move the start back only as far as it takes for every label to clear
    every gate (30+ positives, 10+ rankable pairs at 10 and at 20), and
    never so far that the holdout holds more than 30% of the rows. A label
    that cannot clear the gates within that limit does not move the start;
    its short metrics are reported with their n, flagged.

Every label, because F3 (item 6) has not yet picked the one that ships,
and a rule written around one label would lean the start toward it. 30%
because dev has to keep enough rows to train every experiment on the fix
list and to give the stability sweep test halves of its own; 20-30% is the
usual share for a test set, and today's holdout is 20%.

COUNTED AFTER THE FIXES THAT DELETE ROWS. Two items on the fix list remove
rows, and both land after the freeze:

    F1   version strings (item 2, next)
    F16  PARAMETER_MOVED rows that echo a removal or a required addition
         on the same symbol (item 8)

Counted on today's rows, the holdout could clear the gates now and fall
under them when those fixes land, when it is too late to move. Every other
item changes features, models or flags, not which rows exist.

PERFORMANCE-BLIND BY CONSTRUCTION. Labels are read only to count positives
and rankable pairs, which ml/holdout.py allows ("counting the holdout's
positives to know whether it can be measured is fine") and build.py
already does. It never imports the metrics or the fitting code, so no
model's score on any row can reach the decision. "Rankable" is the
definition in ml/model/metrics.py: a pair with at least one positive and
more than k changes. It is counted here without importing that file, so
the "as built" line has to match what build.py printed.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import version_strings  # noqa: E402
from ml.holdout import (FROZEN_ON, GROUP, HOLDOUT_FILE,  # noqa: E402
                        HOLDOUT_START, MIN_POSITIVES, MIN_RANKABLE_PAIRS,
                        released)

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"
LEDGER = DATA / "holdout_ledger.csv"
LABELS = ("label", "label_scoped", "label_alias")
MAX_SHARE = 0.30

# The kinds a PARAMETER_MOVED row echoes (NOTES F16): griffe reports every
# parameter after a removed or inserted one as moved.
ECHO_OF = {"OBJECT_REMOVED", "PARAMETER_REMOVED", "PARAMETER_ADDED_REQUIRED"}


def load() -> pd.DataFrame:
    for f in (FEATURES, HOLDOUT_FILE):
        if not f.exists():
            sys.exit(f"{f} not found. Run python ml/features/build.py first.")
    keys = {c: str for c in GROUP}
    return pd.concat([pd.read_csv(FEATURES, dtype=keys),
                      pd.read_csv(HOLDOUT_FILE, dtype=keys)],
                     ignore_index=True)


def after_fixes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The rows as built, after F1, and after F1 and F16.

    Version strings are found by the rule build.py uses, not read from a
    column: once F1 runs in build.py the column is gone, and this then
    reports that F1 deletes nothing more.
    """
    version = version_strings(df)
    sibling = (df["kind"].isin(ECHO_OF)
               .groupby([df[c] for c in GROUP + ["symbol"]], dropna=False)
               .transform("any"))
    echo = df["kind"].eq("PARAMETER_MOVED") & sibling.astype(bool)
    return {"as built": df, "after F1": df[~version],
            "after F1 + F16": df[~version & ~echo]}


def pair_table(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """One row per version pair: its day, its size, positives per label.

    Grouped the way holdout_mask groups (missing keys kept), so membership
    matches build.py. `keyed` marks pairs with every key present, because
    metrics._rankable groups with missing keys dropped and never ranks
    those.
    """
    x = df.assign(_when=released(df))
    p = (x.groupby(GROUP, sort=False, dropna=False)
         .agg(day=("_when", "max"), rows=("_when", "size"),
              **{lab: (lab, "sum") for lab in labels})
         .reset_index())
    p["day"] = p["day"].dt.normalize()
    p["keyed"] = p[GROUP].notna().all(axis=1)
    return p


def counts(p: pd.DataFrame, start: pd.Timestamp, labels: list[str]) -> dict:
    """What a holdout of every pair from `start` onward would hold."""
    h = p[p["day"] >= start]
    out = {"start": start, "rows": int(h["rows"].sum()), "pairs": len(h),
           "packages": int(h["package"].nunique())}
    for lab in labels:
        has_pos = h[lab] > 0
        out[lab] = (int(h[lab].sum()),
                    int((h["keyed"] & has_pos & (h["rows"] > 10)).sum()),
                    int((h["keyed"] & has_pos & (h["rows"] > 20)).sum()))
    return out


def clears(c: dict, lab: str) -> bool:
    pos, r10, r20 = c[lab]
    return (pos >= MIN_POSITIVES and r10 >= MIN_RANKABLE_PAIRS
            and r20 >= MIN_RANKABLE_PAIRS)


def fmt(c: dict, lab: str) -> str:
    return "{}/{}/{}".format(*c[lab])


def main() -> None:
    every = load()
    labels = [lab for lab in LABELS if lab in every]
    views = after_fixes(every)
    p_now = pair_table(every, labels)
    fixed = views["after F1 + F16"]
    p = pair_table(fixed, labels)
    total = len(fixed)

    now = counts(p_now, HOLDOUT_START, labels)
    print("\nHOLDOUT START, counts only: nothing is fitted or scored.\n")
    print(f"  {len(every):,} rows in {len(p_now):,} version pairs; the "
          f"holdout frozen {FROZEN_ON} starts {HOLDOUT_START.date()}")
    print(f"  gates: {MIN_POSITIVES}+ positives, {MIN_RANKABLE_PAIRS}+ "
          "rankable pairs at 10 and at 20")
    print("  each label reads  positives / rankable at 10 / rankable at 20")

    head = "".join(f"{lab:>15}" for lab in labels)
    print(f"\n  from {HOLDOUT_START.date()}            rows  pairs{head}")
    for name, rows in views.items():
        c = counts(pair_table(rows, labels), HOLDOUT_START, labels)
        cells = "".join(f"{fmt(c, lab):>15}" for lab in labels)
        print(f"    {name:<22}{c['rows']:>7,}{c['pairs']:>7,}{cells}")
    print("  The 'as built' line has to match build.py's. If it does not, "
          "stop there.")
    print(f"\n  F1 deletes {len(every) - len(views['after F1']):,} rows and "
          f"F16 {len(views['after F1']) - total:,} more. Everything below "
          "is counted\n  after both, on "
          f"{total:,} rows.")

    # Every day a pair was released, newest first. Moving the start back one
    # such day adds that day's pairs, so each count can only grow, and the
    # first day a label clears the gates is the latest start that works.
    days = sorted({d for d in p["day"].dropna() if d < HOLDOUT_START},
                  reverse=True)
    table = []
    for start in [HOLDOUT_START] + days:
        c = counts(p, start, labels)
        c["share"] = c["rows"] / total if total else 0.0
        if table and c["share"] > MAX_SHARE:
            break
        table.append(c)

    print(f"\n  How the holdout grows as its start moves back (up to "
          f"{MAX_SHARE:.0%} of the rows):")
    print(f"    {'start':<12}{'rows':>7}{'share':>7}{'pairs':>7}{head}")
    week = HOLDOUT_START
    for c in table:
        if c["start"] > week:
            continue
        cells = "".join(f"{fmt(c, lab):>15}" for lab in labels)
        print(f"    {str(c['start'].date()):<12}{c['rows']:>7,}"
              f"{c['share']:>7.1%}{c['pairs']:>7,}{cells}")
        week = c["start"] - pd.Timedelta(days=7)
    limit = table[-1]

    print("\n  The latest start at which each label clears every gate:")
    latest = {}
    for lab in labels:
        hit = next((c for c in table if clears(c, lab)), None)
        latest[lab] = hit
        if hit is None:
            print(f"    {lab:<14} none within {MAX_SHARE:.0%}. At the limit "
                  f"({limit['start'].date()}) it has {fmt(limit, lab)}.")
        elif hit["start"] == HOLDOUT_START:
            print(f"    {lab:<14} {HOLDOUT_START.date()}, the frozen start: "
                  "it clears them already")
        else:
            print(f"    {lab:<14} {hit['start'].date()}   holdout "
                  f"{hit['share']:.1%} of the rows")

    if LEDGER.exists():
        print(f"\n  {LEDGER} exists: the holdout has been opened, so its "
              "start can no\n  longer move. Whatever falls short above is "
              "reported with its n, flagged.")
        return

    reached = [c["start"] for c in latest.values() if c is not None]
    start = min(reached) if reached else HOLDOUT_START
    short = [lab for lab, c in latest.items() if c is None]
    at = next(c for c in table if c["start"] == start)
    cells = "   ".join(f"{lab} {fmt(at, lab)}" for lab in labels)

    if start == HOLDOUT_START:
        print(f"\n  VERDICT: KEEP {HOLDOUT_START.date()}.")
        if not short:
            print("  After F1 and F16 every label still clears every gate.")
    else:
        before = counts(p_now, start, labels)
        print(f"\n  VERDICT: MOVE the start from {HOLDOUT_START.date()} to "
              f"{start.date()}.")
        print(f"    after F1 + F16: {at['rows']:,} rows ({at['share']:.1%}), "
              f"{at['pairs']:,} pairs, {at['packages']:,} packages; dev "
              f"keeps {total - at['rows']:,}")
        print(f"    {cells}")
        print(f"    as built today: {before['rows']:,} rows, "
              f"{before['pairs']:,} pairs (now {now['rows']:,} and "
              f"{now['pairs']:,})")
    if short:
        print(f"  Cannot clear the gates within {MAX_SHARE:.0%} of the rows: "
              f"{', '.join(short)}.\n  That does not move the start. Their "
              "metrics below a gate are reported\n  with their n, flagged.")
    print("\n  This script changes nothing. The start is HOLDOUT_START in "
          "ml/holdout.py.")


if __name__ == "__main__":
    main()
