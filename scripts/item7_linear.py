"""
Item 7 of the fix list, first half (F8): does a straight line do what
the trees do? Decided by the rule in NOTES §29.1, written before this
was first run.

    python scripts/item7_linear.py

Reads  data/features.csv
Writes data/item7_linear.csv, both sweeps row by row. Nothing else is
       touched: no model, and no stability file that train.py reads.

WHAT IS COMPARED. Three models on the same rows at each of the seven cut
dates of §23.5, each fitted on everything before the date and scored on
what follows:

  line        a logistic regression on the ranker's 17 features, fitted
              on the same rows (baselines.linear_scores). Nothing tuned.
  classifier  LightGBM with the binary objective, on the same features,
              its tree count chosen by the same CV. Trees, pointwise.
  ranker      LightGBM lambdarank, the model that ships (§24).

The report's claim (F8) is that the trees find a U-shape a line cannot.
The like-for-like test of that is the classifier against the line: both
score each change on its own, so the only difference is trees against a
line. The ranker is trained to order changes INSIDE an upgrade and never
to compare them across upgrades, and pooled PR-AUC grades exactly that
comparison (§21.9, F14 and F25), so ranker-against-line mixes two
questions. It is printed beside the other because the ranker is what the
report describes, and it is read on nDCG@20 as well, the within-upgrade
measure the ranker is fitted for.

IT CHECKS ITSELF. The lambdarank sweep has to reproduce §24.2's
label_alias column lift for lift, which §27.2 reproduced on 1 Oct. If it
does not, the "ranker" here is not the model that ships, and no verdict
is printed. And the line is fitted twice, once in each sweep, on the
same rows: the two must agree at every date, or the sweeps did not see
the same data.

THE RULE (NOTES §29.1), for the classifier against the line, on PR-AUC.
At each date r = classifier PR-AUC / line PR-AUC.

  trees win    r > 1 at 6 or more of the 7 dates AND the median r is
               1.25 or more. The claim stays, as evidence, with the
               numbers.
  line wins    r < 1 at 6 or more of the 7 dates AND the median r is
               0.80 or less. The claim is withdrawn, and whether the line
               should ship becomes a question (item 15, F25).
  no difference   anything else. The claim is withdrawn; the result is in
               the features, not in the trees.

A ratio, not a difference, because PR-AUC's floor is the positive rate
and the floor differs across dates (0.028 to 0.086 in §27.2); on one
date's test half the two numbers share a floor and the ratio is on one
scale. 1.25 is a quarter, §24.1's and §25.3's margin, as a ratio. 6 of
7, as in §25.3: a coin comes up 6 or more of 7 about one time in
sixteen, and the dates share most of their data, so that is a floor on
how often chance could do it, not a p-value.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import assert_no_holdout  # noqa: E402
from ml.model import stability  # noqa: E402
from ml.model.train import prepare  # noqa: E402

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"
OUT = DATA / "item7_linear.csv"
LABEL = "label_alias"

# The cut dates of §23.5, where §24 measured the model that ships.
DATES = ["2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
         "2026-03-02", "2026-04-02", "2026-05-04"]
# §24.2, label_alias: its lift over popularity at those dates, reproduced
# in §27.2. The lambdarank sweep below has to give exactly these.
SHIPPED = [4.42, 4.86, 5.70, 4.02, 3.01, 4.08, 2.30]

MARGIN = 1.25   # the median ratio trees need, and 1/MARGIN the line
NEED = 6        # dates of the 7 a side must win


def verdict(trees: list[float], line: list[float]) -> dict:
    """§29.1 applied to two PR-AUC columns, one value per date, in the
    same order. Returns the outcome ("trees", "line" or "none") with the
    numbers it rests on."""
    if len(trees) != len(line):
        raise ValueError("one PR-AUC per date on each side")
    ratios = [t / l_ for t, l_ in zip(trees, line)]
    n = len(ratios)
    wins = sum(r > 1 for r in ratios)
    losses = sum(r < 1 for r in ratios)
    median = float(pd.Series(ratios).median())
    # PR-AUCs are stored to 4 places, so a median that reads exactly
    # 1.25x or 0.80x counts, and floating point does not get a vote.
    if wins >= NEED and median >= MARGIN - 1e-9:
        outcome = "trees"
    elif losses >= NEED and median <= 1 / MARGIN + 1e-9:
        outcome = "line"
    else:
        outcome = "none"
    return {"outcome": outcome, "n": n, "wins": wins, "losses": losses,
            "median": median, "ratios": ratios}


def main() -> None:
    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found. Run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    assert_no_holdout(df, "item7_linear.py")
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    runs = {}
    for objective in ("lambdarank", "binary"):
        print(f"\n{objective}: fitting at {len(DATES)} cut dates ...",
              flush=True)
        t = stability.run_label(df, LABEL, feats, objective, "cv", DATES)
        runs[objective] = t.set_index("cut").reindex(DATES)
    rk, clf = runs["lambdarank"], runs["binary"]

    print(f"\n{LABEL.upper()} AT THE SEVEN DATES OF §23.5: THE LINE, THE "
          "CLASSIFIER, THE RANKER\n")
    got = [None if s else round(float(v), 2)
           for s, v in zip(rk["skipped"].astype(bool), rk["lift_vs_pop"])]
    reproduced = got == SHIPPED
    if reproduced:
        print("  self-check: the ranker reproduces §24.2, lift for lift")
    else:
        print("  self-check: the ranker DOES NOT reproduce §24.2")
        print(f"      §24.2    {SHIPPED}\n      this run {got}")
    usable = (~rk["skipped"].astype(bool) & ~clf["skipped"].astype(bool))
    agree = all(abs(float(rk.loc[d, "linear"]) - float(clf.loc[d, "linear"]))
                < 1e-4 for d in DATES if usable[d])
    if agree:
        print("  self-check: the line fitted in each sweep agrees at every "
              "date")
    else:
        print("  self-check: the line DIFFERS between the two sweeps, so "
              "they did not\n      see the same rows. No verdict.")

    def cell(t: pd.DataFrame, d: str, col: str, fmt: str) -> str:
        v = t.loc[d, col]
        return "-" if bool(t.loc[d, "skipped"]) or pd.isna(v) else fmt % v

    print(f"\n  {'':<12}{'PR-AUC':^32}{'over the line':^22}"
          f"{'nDCG@20':^18}")
    print(f"  {'cut':<12}{'floor':>8}{'line':>8}{'classif':>8}"
          f"{'ranker':>8}{'classif':>11}{'ranker':>11}"
          f"{'line':>9}{'ranker':>9}")
    for d in DATES:
        print(f"  {d:<12}{cell(rk, d, 'floor', '%.4f'):>8}"
              f"{cell(rk, d, 'linear', '%.4f'):>8}"
              f"{cell(clf, d, 'pr_auc', '%.4f'):>8}"
              f"{cell(rk, d, 'pr_auc', '%.4f'):>8}"
              f"{cell(clf, d, 'lift_vs_linear', '%.2fx'):>11}"
              f"{cell(rk, d, 'lift_vs_linear', '%.2fx'):>11}"
              f"{cell(rk, d, 'linear_ndcg_20', '%.4f'):>9}"
              f"{cell(rk, d, 'ndcg_20', '%.4f'):>9}")

    def stat(t: pd.DataFrame, col: str, how: str) -> float:
        v = t.loc[~t["skipped"].astype(bool), col].astype(float)
        return float(getattr(v, how)())

    print(f"  {'median':<12}{'':>8}{stat(rk, 'linear', 'median'):>8.4f}"
          f"{stat(clf, 'pr_auc', 'median'):>8.4f}"
          f"{stat(rk, 'pr_auc', 'median'):>8.4f}"
          f"{stat(clf, 'lift_vs_linear', 'median'):>10.2f}x"
          f"{stat(rk, 'lift_vs_linear', 'median'):>10.2f}x"
          f"{stat(rk, 'linear_ndcg_20', 'median'):>9.4f}"
          f"{stat(rk, 'ndcg_20', 'median'):>9.4f}")

    print("\n  Lift over popularity, for the record:")
    print(f"  {'':<12}{'line':>10}{'classifier':>12}{'ranker':>10}")
    for d in DATES:
        if not usable[d]:
            continue
        pop = float(rk.loc[d, "popularity"])
        print(f"  {d:<12}{float(rk.loc[d, 'linear']) / pop:>9.2f}x"
              f"{float(clf.loc[d, 'lift_vs_pop']):>11.2f}x"
              f"{float(rk.loc[d, 'lift_vs_pop']):>9.2f}x")
    print(f"  trees chosen by CV: classifier {int(stat(clf, 'trees', 'min'))}"
          f"–{int(stat(clf, 'trees', 'max'))}, ranker "
          f"{int(stat(rk, 'trees', 'min'))}–{int(stat(rk, 'trees', 'max'))}"
          " (20 is the clamp, F7)")

    print("\nTHE RULE, NOTES §29.1, fixed before this run\n")
    if not reproduced or not agree:
        print("  No verdict. Find out why the self-check failed before "
              "reading anything above.")
    elif int(usable.sum()) != len(DATES):
        print(f"  No verdict. Only {int(usable.sum())} of the {len(DATES)} "
              "dates were measured by both; the rule was\n  written for "
              f"all {len(DATES)}.")
    else:
        v = verdict(clf["pr_auc"].astype(float).tolist(),
                    rk["linear"].astype(float).tolist())
        print(f"  like for like, the classifier against the line, PR-AUC:")
        print(f"     classifier ahead at {v['wins']}/{v['n']} dates, behind "
              f"at {v['losses']}/{v['n']}; median ratio {v['median']:.4f}x")
        # Four places, not two. On 2 Oct the median was 1.2499 and this
        # line printed "1.25x" above a verdict that it fell short of
        # 1.25, which reads as a bug. The rule reads the value, not the
        # print (NOTES §29.2).
        print(f"     trees win: ahead at {NEED}+ and median >= {MARGIN:.2f}x"
              f"     line wins: behind at {NEED}+ and median <= "
              f"{1 / MARGIN:.2f}x")
        w = verdict(rk["pr_auc"].astype(float).tolist(),
                    rk["linear"].astype(float).tolist())
        nd = int((rk["ndcg_20"].astype(float) > rk["linear_ndcg_20"]
                  .astype(float)).sum())
        print(f"  and the ranker that ships against the line: ahead at "
              f"{w['wins']}/{w['n']} on PR-AUC (median ratio "
              f"{w['median']:.2f}x),\n     ahead at {nd}/{w['n']} on "
              "nDCG@20")
        if v["outcome"] == "trees":
            print("\n  VERDICT: the trees beat the line. The claim stays, "
                  "with these numbers (NOTES §29.2).")
        elif v["outcome"] == "line":
            print("\n  VERDICT: the line beats the trees. The claim is "
                  "withdrawn, and item 15 (F25)\n  decides what ships "
                  "(NOTES §29.2).")
        else:
            print("\n  VERDICT: no difference the rule can see. The claim "
                  "is withdrawn: the result is\n  in the features, not "
                  "the trees (NOTES §29.2).")

    both = pd.concat([rk.reset_index().assign(objective="lambdarank"),
                      clf.reset_index().assign(objective="binary")],
                     ignore_index=True)
    both.to_csv(OUT, index=False)
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
