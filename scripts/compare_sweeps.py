"""
Before and after a fix, at the same cut dates: two stability sweeps, side
by side.

    python scripts/compare_sweeps.py BEFORE.csv AFTER.csv

Reads  two files ml/model/stability.py wrote. The "after" one is usually
       a `--at` run at the "before" one's dates (data/stability_..._at.csv)
Writes nothing

WHY. A fix that changes which rows exist moves the sweep's quantile dates,
and two sweeps at different dates are two different exams (§15, §23.2).
So this refuses to compare unless both files measured the same cut dates
and were run the same way (holdout boundary, relevance, tuning).

HOW TO READ IT (§23.2, §24.1). At each date, the lift over popularity
before -> after, and the numbers under it. The sweep cannot tell apart
two lifts within 0.25x of each other (§24.1), so a date counts as moved
only by more than that; the last line says how many did, and which way.
"""

import argparse
import pathlib
import sys

import pandas as pd

STAMPS = ("holdout_from", "relevance", "tuning")
RESOLUTION = 0.25   # §24.1: what the sweep cannot resolve


def usable(t: pd.DataFrame) -> pd.DataFrame:
    """The cuts that were measured, by date. A skipped cut (too few
    positives, or a repeat of another split) has no numbers."""
    return t[~t["skipped"].astype(bool)].set_index("cut")


def stamp(t: pd.DataFrame, col: str) -> list[str]:
    return sorted(t[col].astype(str).unique()) if col in t else ["(none)"]


def refusal(before: pd.DataFrame, after: pd.DataFrame) -> str | None:
    """Why the two cannot be compared, or None."""
    for col in STAMPS:
        if stamp(before, col) != stamp(after, col):
            return (f"The two sweeps were not run the same way: {col} is "
                    f"{', '.join(stamp(before, col))} before and "
                    f"{', '.join(stamp(after, col))} after.")
    b, a = usable(before), usable(after)
    if set(b.index) != set(a.index):
        only_b = sorted(set(b.index) - set(a.index))
        only_a = sorted(set(a.index) - set(b.index))
        return ("Different cut dates, so different exams.\n"
                f"  measured only before: {', '.join(only_b) or 'none'}\n"
                f"  measured only after:  {', '.join(only_a) or 'none'}\n"
                "Run the after sweep at the before sweep's dates:\n"
                "  python ml/model/stability.py --at "
                + ",".join(sorted(b.index)))
    if b.empty:
        return "Neither sweep measured any cut."
    return None


def moved(b: float, a: float) -> str:
    """"up" or "down" if the lift moved by more than the sweep resolves.
    Lifts are stored to two places, so the difference is rounded to two
    and a move of exactly 0.25x is not counted, whatever floating point
    makes of 5.45 - 5.20."""
    d = round(float(a) - float(b), 2)
    return "up" if d > RESOLUTION else "down" if d < -RESOLUTION else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Two stability sweeps at the "
                                 "same cut dates, before and after a fix.")
    ap.add_argument("before", help="the sweep before the fix")
    ap.add_argument("after", help="the sweep after it, at the same dates")
    args = ap.parse_args()
    for f in (args.before, args.after):
        if not pathlib.Path(f).exists():
            sys.exit(f"{f} not found.")
    before = pd.read_csv(args.before, dtype={"cut": str})
    after = pd.read_csv(args.after, dtype={"cut": str})
    why = refusal(before, after)
    if why:
        sys.exit(why)
    b, a = usable(before), usable(after)
    dates = sorted(b.index)
    n = len(dates)

    print(f"\nBEFORE {args.before}\nAFTER  {args.after}")
    print(f"  {n} cut dates measured by both; tuning "
          f"{', '.join(stamp(after, 'tuning'))}, relevance "
          f"{', '.join(stamp(after, 'relevance'))}\n")
    print(f"  {'cut':<12}{'lift over pop':<16}{'moved':<7}{'PR-AUC':<18}"
          f"{'test rows':<16}nDCG@20")
    for d in dates:
        x, y = b.loc[d], a.loc[d]
        lift = f"{x['lift_vs_pop']:.2f}x -> {y['lift_vs_pop']:.2f}x"
        pr = f"{x['pr_auc']:.4f} -> {y['pr_auc']:.4f}"
        rows = f"{int(x['test_rows']):,} -> {int(y['test_rows']):,}"
        nd = f"{x['ndcg_20']:.4f} -> {y['ndcg_20']:.4f}"
        mark = moved(x["lift_vs_pop"], y["lift_vs_pop"])
        print(f"  {d:<12}{lift:<16}{mark:<7}{pr:<18}{rows:<16}{nd}")

    def both(col: str, how: str, fmt: str) -> str:
        bv = getattr(b[col].astype(float), how)()
        av = getattr(a[col].astype(float), how)()
        return f"{fmt % bv} -> {fmt % av}"

    print(f"\n  lift over popularity   median "
          f"{both('lift_vs_pop', 'median', '%.2fx')}, worst "
          f"{both('lift_vs_pop', 'min', '%.2fx')}")
    print(f"  beats popularity       {int(b['beats_pop'].sum())}/{n} -> "
          f"{int(a['beats_pop'].sum())}/{n}")
    if "lift_lo" in b and "lift_lo" in a:
        print(f"  interval above 1.0x    {int((b['lift_lo'] > 1).sum())}/{n}"
              f" -> {int((a['lift_lo'] > 1).sum())}/{n}")
    if "beats_linear" in b and "beats_linear" in a:
        print(f"  beats the line         {int(b['beats_linear'].sum())}/{n} "
              f"-> {int(a['beats_linear'].sum())}/{n}")
    print(f"  PR-AUC                 median "
          f"{both('pr_auc', 'median', '%.4f')}")
    print(f"  nDCG@20                median "
          f"{both('ndcg_20', 'median', '%.4f')}")
    print(f"  precision@10           median "
          f"{both('p_at_10', 'median', '%.3f')}")

    marks = [moved(b.loc[d, "lift_vs_pop"], a.loc[d, "lift_vs_pop"])
             for d in dates]
    up, down = marks.count("up"), marks.count("down")
    if up or down:
        print(f"\n  moved by more than {RESOLUTION}x: {up + down} of {n} "
              f"dates ({up} up, {down} down)")
    else:
        print(f"\n  moved by more than {RESOLUTION}x: none of the {n} "
              "dates. The fix moved no lift by more than the sweep can "
              "resolve.")


if __name__ == "__main__":
    main()
