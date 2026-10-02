"""
Item 7 of the fix list, second half (F7): should the trees be sized by
the data instead of a constant? Decided by the rule in NOTES §30.1,
written before this was first run.

    python scripts/item7_tuning.py

Reads  data/features.csv
Writes data/item7_tuning.csv, both sweeps row by row. Nothing else is
       touched: no model, and no stability file that train.py reads.

WHAT IS COMPARED. Two ways of sizing the same ranker (label_alias,
lambdarank, binary relevance, the 17 features), at the seven cut dates
of §23.5, each fitted on everything before a date:

  fixed   what ships. One setting (31 leaves, learning rate 0.05,
          smallest leaf 30). Its tree count is the median of four CV
          folds that stop on LightGBM's nDCG@10, then clamped at 20.
  tuned   --tuning cv. 18 settings (train.GRID) tried on the same four
          folds, each fold stopping on PR-AUC in its window; the setting
          with the best median PR-AUC over its folds' floors wins, and its
          tree count is the median of its folds. No clamp.

Everything is chosen inside each date's training rows. No test row and
no holdout row is seen by any choice.

IT CHECKS ITSELF. The fixed sweep has to reproduce §24.2's label_alias
column lift for lift (as §27.2 and §29.2 did). If it does not, "fixed"
is not the model that ships, and no verdict is printed.

THE RULE (NOTES §30.1). Tuned ships only if all three hold:

  1. It beats popularity at all 7 dates.
  2. Its worst lift over popularity is at most 0.25x below fixed's.
  3. Its median lift is at most 0.50x below fixed's.

Otherwise fixed stays. 0.25x is §24.1's margin for worst cases, 0.50x
stability.py's for medians: two models closer than that are more than
the sweep can tell apart. Tuned does not have to win, because the thing
F7 fixes is not a score: fixed's size is set by a constant, 20 trees,
and a model whose size the data chose is the one to defend in a viva.
It has to cost nothing the sweep can see.

Printed beside the rule and not in it: how often each fold's validation
window has upgrades with no positive (F7's diagnosis), the setting each
date chose and how often it sat at the edge of the grid, and the tuned
model against the line. F8's verdict (§29.2) was about the shipped trees
and is not reopened by this.
"""

import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import GROUP, assert_no_holdout  # noqa: E402
from ml.model import stability  # noqa: E402
from ml.model.train import cv_folds, prepare  # noqa: E402

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"
OUT = DATA / "item7_tuning.csv"
LABEL = "label_alias"

# The cut dates of §23.5, where §24 measured the model that ships.
DATES = ["2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
         "2026-03-02", "2026-04-02", "2026-05-04"]
# §24.2, label_alias: its lift over popularity at those dates. The fixed
# sweep below has to give exactly these.
SHIPPED = [4.42, 4.86, 5.70, 4.02, 3.01, 4.08, 2.30]

WORST_MARGIN = 0.25    # §24.1
MEDIAN_MARGIN = 0.50   # stability.py's "median lift does NOT separate"
# The grid's two corners: the simplest trees it offers, and the most
# flexible. A setting chosen at a corner at most dates says the best one
# may lie beyond the grid in that direction.
SIMPLEST = (7, 100)     # (leaves, smallest leaf)
FLEXIBLE = (31, 10)


def verdict(fixed: list[float], tuned: list[float],
            tuned_beats: int) -> tuple[bool, list[str]]:
    """§30.1 applied to the two sweeps' lifts over popularity, one value
    per date in each, and the number of dates tuned beats popularity at.
    Returns (does tuned ship, the reasoning line by line)."""
    if len(fixed) != len(tuned):
        raise ValueError("one lift per date on each side")
    n = len(fixed)
    worst_f, worst_t = min(fixed), min(tuned)
    med_f, med_t = float(np.median(fixed)), float(np.median(tuned))
    # Lifts are stored to 2 places: exactly at a margin still holds.
    floor_w = round(worst_f - WORST_MARGIN, 2)
    floor_m = round(med_f - MEDIAN_MARGIN, 2)
    beats = tuned_beats == n
    worst_ok = worst_t >= floor_w - 1e-9
    median_ok = med_t >= floor_m - 1e-9

    def line(text: str, ok: bool, value: str) -> str:
        return f"  {text:<56}{'yes' if ok else 'no'}, {value}"

    lines = [
        line("1. beats popularity at every date", beats,
             f"{tuned_beats}/{n}"),
        line(f"2. worst lift at least {floor_w:.2f}x (fixed's "
             f"{worst_f:.2f}x - {WORST_MARGIN}x)", worst_ok,
             f"{worst_t:.2f}x"),
        line(f"3. median lift at least {floor_m:.2f}x (fixed's "
             f"{med_f:.2f}x - {MEDIAN_MARGIN}x)", median_ok,
             f"{med_t:.2f}x"),
    ]
    return beats and worst_ok and median_ok, lines


def empty_share(df: pd.DataFrame) -> list[tuple[int, int]]:
    """For each CV fold of these rows: (upgrades in its validation window,
    how many of them have no positive). The number fixed's folds stop on
    gives every one of the second kind nDCG 1.0, whatever the model does."""
    out = []
    for _, val in cv_folds(df.sort_values(GROUP), LABEL):
        per = val.groupby(GROUP, observed=True)[LABEL].sum()
        out.append((len(per), int((per == 0).sum())))
    return out


def main() -> None:
    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found. Run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    assert_no_holdout(df, "item7_tuning.py")
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    print("\nWHY F7 CHANGES THE STOPPING NUMBER: the four CV folds of all "
          "dev rows")
    for i, (n, empty) in enumerate(empty_share(df), 1):
        print(f"  fold {i}: {empty:,} of {n:,} validation upgrades have no "
              f"positive ({empty / n:.0%}), each scored nDCG 1.0 by "
              "LightGBM")

    runs, took = {}, {}
    for tuning in ("fixed", "cv"):
        print(f"\n{tuning}: fitting at {len(DATES)} cut dates ...",
              flush=True)
        t0 = time.time()
        t = stability.run_label(df, LABEL, feats, "lambdarank", "cv",
                                DATES, tuning=tuning)
        took[tuning] = time.time() - t0
        runs[tuning] = t.set_index("cut").reindex(DATES)
    fx, tu = runs["fixed"], runs["cv"]
    print(f"  (fixed {took['fixed'] / 60:.1f} min, tuned "
          f"{took['cv'] / 60:.1f} min)")

    print(f"\n{LABEL.upper()} AT THE SEVEN DATES OF §23.5, FIXED AGAINST "
          "TUNED\n")
    got = [None if s else round(float(v), 2)
           for s, v in zip(fx["skipped"].astype(bool), fx["lift_vs_pop"])]
    reproduced = got == SHIPPED
    if reproduced:
        print("  self-check: fixed reproduces §24.2, lift for lift")
    else:
        print("  self-check: fixed DOES NOT reproduce §24.2")
        print(f"      §24.2    {SHIPPED}\n      this run {got}")

    def cell(t: pd.DataFrame, d: str, col: str, fmt: str) -> str:
        v = t.loc[d, col]
        return "-" if bool(t.loc[d, "skipped"]) or pd.isna(v) else fmt % v

    print(f"\n  {'':<12}{'lift over popularity':^20}{'trees':^12}"
          f"{'':>13}{'nDCG@20':^18}{'over the line':^14}")
    print(f"  {'cut':<12}{'fixed':>10}{'tuned':>10}{'fixed':>6}"
          f"{'tuned':>6}{'tuned setting':>15}{'fixed':>9}{'tuned':>9}"
          f"{'tuned':>12}")
    for d in DATES:
        print(f"  {d:<12}{cell(fx, d, 'lift_vs_pop', '%.2fx'):>10}"
              f"{cell(tu, d, 'lift_vs_pop', '%.2fx'):>10}"
              f"{cell(fx, d, 'trees', '%d'):>6}"
              f"{cell(tu, d, 'trees', '%d'):>6}"
              f"{cell(tu, d, 'setting', '%s'):>15}"
              f"{cell(fx, d, 'ndcg_20', '%.4f'):>9}"
              f"{cell(tu, d, 'ndcg_20', '%.4f'):>9}"
              f"{cell(tu, d, 'lift_vs_linear', '%.2fx'):>12}")

    def stat(t: pd.DataFrame, col: str, how: str) -> float:
        v = t.loc[~t["skipped"].astype(bool), col].astype(float)
        return float(getattr(v, how)())

    print(f"  {'worst':<12}{stat(fx, 'lift_vs_pop', 'min'):>9.2f}x"
          f"{stat(tu, 'lift_vs_pop', 'min'):>9.2f}x")
    print(f"  {'median':<12}{stat(fx, 'lift_vs_pop', 'median'):>9.2f}x"
          f"{stat(tu, 'lift_vs_pop', 'median'):>9.2f}x"
          f"{'':>27}{stat(fx, 'ndcg_20', 'median'):>9.4f}"
          f"{stat(tu, 'ndcg_20', 'median'):>9.4f}"
          f"{stat(tu, 'lift_vs_linear', 'median'):>11.2f}x")

    print("\n  The other numbers, for the record (medians over the dates):")
    print(f"  {'':<12}{'PR-AUC':>10}{'precision@10':>15}{'nDCG@20':>10}")
    for name, t in (("fixed", fx), ("tuned", tu)):
        print(f"  {name:<12}{stat(t, 'pr_auc', 'median'):>10.4f}"
              f"{stat(t, 'p_at_10', 'median'):>15.4f}"
              f"{stat(t, 'ndcg_20', 'median'):>10.4f}")

    usable = ~tu["skipped"].astype(bool)
    settings = tu.loc[usable, "setting"].astype(str)
    corner = {"simplest": 0, "flexible": 0}
    for s in settings:
        leaves, _lr, smallest = s.split("/")
        shape = (int(leaves), int(smallest))
        corner["simplest"] += shape == SIMPLEST
        corner["flexible"] += shape == FLEXIBLE
    print("\n  settings chosen (leaves/learning rate/smallest leaf): "
          + ", ".join(f"{k} x{v}" for k, v in
                      settings.value_counts().items()))
    for name, shape in (("simplest", SIMPLEST), ("flexible", FLEXIBLE)):
        n = corner[name]
        print(f"  at the grid's {name} corner ({shape[0]} leaves, smallest "
              f"leaf {shape[1]}) at {n}/{len(settings)} dates"
              + ("; the best setting may lie beyond the grid that way. "
                 "Recorded, not ruled on." if n >= 4 else ""))

    print("\nTHE RULE, NOTES §30.1, fixed before this run\n")
    if not reproduced:
        print("  No verdict. The fixed sweep is not the model §24 measured,")
        print("  so the comparison is not the one the rule was written for.")
    elif int((~fx["skipped"].astype(bool) & usable).sum()) != len(DATES):
        print("  No verdict. A date was not measured by both sweeps, and "
              f"the rule was\n  written for all {len(DATES)}.")
    else:
        ships, lines = verdict(fx["lift_vs_pop"].astype(float).tolist(),
                               tu["lift_vs_pop"].astype(float).tolist(),
                               int(tu["beats_pop"].astype(bool).sum()))
        for line in lines:
            print(line)
        if ships:
            print("\n  VERDICT: tuned ships. The trees are sized by the data "
                  "from now on (NOTES §30.2).")
        else:
            print("\n  VERDICT: fixed stays. Tuning is kept as an option and "
                  "not shipped (NOTES §30.2).")

    both = pd.concat([fx.reset_index().assign(tuning="fixed"),
                      tu.reset_index().assign(tuning="cv")],
                     ignore_index=True)
    both.to_csv(OUT, index=False)
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
