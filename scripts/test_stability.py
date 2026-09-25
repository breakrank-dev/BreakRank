"""
Does the stability sweep count each split once?

    python scripts/test_stability.py

No data files and no model fit; a few seconds. The fit inside each cut
(one_split) is swapped for a stand-in that records which cut points were
fitted, so what this tests is the bookkeeping around the fits: which cuts
count, and what the summary says about them.

WHY THIS FILE EXISTS. The first sweep after the holdout froze printed
seven cut dates, two of them 2026-04-05 with the same 2,902 test rows.
The quantiles q=0.75 and q=0.80 had both landed on one crowded day, and
the sweep scored that split twice: "beats popularity at 7/7", median lift
2.01x. Six splits had been measured, and their median is 2.04x.

Three cases:

  1. Cut points that select the same rows: the later one is written as a
     skipped row that names the earlier one, and it is never fitted. One
     repeat lands on the crowded day itself and one lands a week after
     it, in the gap before the next release, so a check on the printed
     date alone would miss the second.
  2. Cut points whose test halves differ by a single row are both kept.
     The check compares rows, not how many there are.
  3. The report counts distinct splits: "6/6", and the median over six.
"""

import contextlib
import io
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.holdout import HOLDOUT_START  # noqa: E402
from ml.model import stability  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []

# The lifts of the 26 Sep sweep, by cut point. 0.80 is the repeat: if it
# were fitted and counted, the median would come out 2.01x, not 2.04x.
LIFTS = {0.55: 2.34, 0.60: 1.91, 0.65: 2.19, 0.70: 2.07, 0.75: 2.01,
         0.80: 2.01, 0.85: 1.96}


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def fixture(crowded: range, n: int = 1002, gap_days: int = 10
            ) -> pd.DataFrame:
    """n rows, one release per row, except the rows in `crowded`, which
    all share one day, 2026-04-05. The next release comes gap_days later.

    With n=1002 a cut point q sits at row q*1001 of the sorted dates, so
    q=0.75 is row 750.75, q=0.80 row 800.8, q=0.85 row 850.85.
    """
    day = pd.Timestamp("2026-04-05")
    start = day - pd.Timedelta(days=crowded.start)
    dates = []
    for i in range(n):
        if i < crowded.start:
            dates.append(start + pd.Timedelta(days=i))
        elif i in crowded:
            dates.append(day)
        else:
            dates.append(day + pd.Timedelta(days=gap_days + i - crowded.stop))
    return pd.DataFrame({
        "package": "p",
        "version_from": [str(i) for i in range(n)],
        "version_to": [str(i + 1) for i in range(n)],
        "released_at": [d.strftime("%Y-%m-%d") for d in dates],
        "label": [int(i % 7 == 0) for i in range(n)],
    })


def sweep(df: pd.DataFrame, cuts=None) -> tuple[pd.DataFrame, list[float]]:
    """run_label with the fit swapped out. Returns (table, cuts fitted)."""
    fitted = []

    def stand_in(df, q, label, feats, objective, stopping="cv"):
        fitted.append(q)
        cutoff, is_test = stability.cut(df, q)
        lift = LIFTS.get(q, 1.5)
        return {"cut": str(cutoff.date()), "q": q,
                "test_rows": int(is_test.sum()),
                "test_pos": int(df.loc[is_test, label].sum()),
                "floor": 0.05, "trees": 20, "rankable10": 12,
                "pr_auc": 0.3, "p_at_10": 0.2, "ndcg_20": 0.6,
                "popularity": round(0.3 / lift, 4), "semver": 0.1,
                "lift_vs_pop": lift, "beats_pop": lift > 1,
                "beats_semver": True, "skipped": False}

    real_fit, real_cuts = stability.one_split, stability.CUTS
    stability.one_split = stand_in
    if cuts is not None:
        stability.CUTS = cuts
    try:
        t = stability.run_label(df, "label", [], "lambdarank")
    finally:
        stability.one_split, stability.CUTS = real_fit, real_cuts
    return t, fitted


def repeats(t: pd.DataFrame) -> pd.DataFrame:
    """The rows written as repeats. Empty, not a crash, if there are none."""
    if "same_as" not in t:
        return t.iloc[0:0].assign(same_as=pd.Series(dtype=float))
    return t[t["same_as"].notna()]


def case_repeats() -> None:
    print("\n1. A CUT POINT THAT REPEATS AN EARLIER SPLIT IS NOT FITTED "
          "OR COUNTED")
    # Rows 740..850 share 2026-04-05. q=0.75 and q=0.80 both land inside
    # them; q=0.85 lands between the last of them and the next release,
    # ten days later, so its cut date prints as 2026-04-13.
    t, fitted = sweep(fixture(range(740, 851)))
    check("only the five distinct splits are fitted",
          fitted, [0.55, 0.60, 0.65, 0.70, 0.75])
    rep = repeats(t).set_index("q")
    check("q=0.80 and q=0.85 are written as repeats of q=0.75",
          rep["same_as"].to_dict(), {0.80: 0.75, 0.85: 0.75})
    check("and both are marked skipped",
          rep["skipped"].tolist(), [True, True])
    check("the second repeat prints a different date: rows were compared, "
          "not dates", rep["cut"].get(0.85), "2026-04-13")
    check("each repeat names the crowded day and how many rows it holds",
          (rep.get("shared_day", pd.Series(dtype=object)).tolist(),
           rep.get("shared_day_rows", pd.Series(dtype=float)).tolist()),
          (["2026-04-05", "2026-04-05"], [111, 111]))
    check("a repeat's test half is the original's, row for row",
          rep["test_rows"].tolist(),
          t.loc[t["q"] == 0.75, "test_rows"].tolist() * 2)
    check("every row, repeats included, carries the holdout_from stamp "
          "train.py requires",
          t["holdout_from"].tolist(), [str(HOLDOUT_START.date())] * 7)


def case_near_repeat() -> None:
    print("\n2. TWO SPLITS THAT DIFFER BY ONE ROW ARE BOTH KEPT")
    # No crowded day, one release a day, so q=0.500 and q=0.501 (rows 500.5
    # and 501.5) give test halves of 501 and 500 rows.
    t, fitted = sweep(fixture(range(0)), cuts=[0.500, 0.501])
    check("both are fitted", fitted, [0.500, 0.501])
    check("neither is a repeat", len(repeats(t)), 0)
    check("their test halves differ by exactly one row",
          t["test_rows"].tolist(), [501, 500])


def case_report() -> None:
    print("\n3. THE REPORT COUNTS SIX SPLITS, NOT SEVEN")
    # Rows 740..805 share the day: q=0.75 and q=0.80 land in it and q=0.85
    # does not, the pattern of the 26 Sep sweep.
    t, fitted = sweep(fixture(range(740, 806)))
    check("one repeat, q=0.80", repeats(t)["q"].tolist(), [0.80])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        stability.report(t, "label")
    text = out.getvalue()
    check("the header says 7 cut points, 6 distinct splits",
          "7 cut points, 6 distinct splits" in text, True)
    check("the repeat is listed as skipped, naming q=0.75",
          "skipped q=0.80 (2026-04-05): the same split as q=0.75" in text,
          True)
    check("beats popularity at 6/6, not 7/7",
          ("beats popularity at 6/6" in text, "7/7" in text), (True, False))
    check("median lift over the six is 2.04x (2.01x if the repeat "
          "counted)", "lift vs pop   2.04x" in text, True)
    if failures:
        print(text)


def main() -> None:
    case_repeats()
    case_near_repeat()
    case_report()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Each split the sweep reports was measured "
          "once.")


if __name__ == "__main__":
    main()
