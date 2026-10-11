"""
Item 3 of the fix list (F2): should the ranker be taught graded relevance?
Decided by the rule in NOTES §25, written before this was first run.

    python scripts/item3_relevance.py

Reads  data/features.csv
Writes data/item3_relevance.csv, both sweeps row by row. Nothing else is
       touched: no model, and no stability file that train.py reads.

WHAT IS COMPARED. Two pipelines that differ in one thing, what LightGBM
is told to put on top of an upgrade:

  binary   every positive is worth the same. This is what ships now.
  graded   a positive is worth 1, 3, 7 or 15 by how many packages use
           it: 1 package, 2 to 6, 7 to 19, 20 or more (train.grades).

Everything else is held still: label_alias, the 17 features, the tree
count chosen by CV, the refitted baselines, and the seven cut dates of
§23.5, where §24 measured the model that ships. Which rows are positive,
and so which cuts are skipped, comes from the 0/1 label, so both sit the
same seven exams.

IT CHECKS ITSELF. The binary sweep has to reproduce §24.2's label_alias
column lift for lift. If it does not, the "binary" here is not the model
that ships, and no verdict is printed.

THE RULE (NOTES §25.3). Graded ships only if both of these hold:

  1. It loses nothing that matters. Its worst lift over popularity is at
     most 0.25x below binary's, and it beats popularity at every date.
  2. It wins clearly somewhere. Its worst lift is more than 0.25x above
     binary's, or its nDCG@20 with graded gains is higher than binary's
     at 6 or more of the 7 dates.

Otherwise binary stays. 0.25x is §24.1's margin: two worst cases closer
than that are more than this sweep can tell apart. 6 of 7, because a
coin would come up 5 or more of 7 about one time in four, and 6 or more
about one time in sixteen. The dates share most of their data, so that
is a floor on how often chance could do it, not a p-value.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import assert_no_holdout  # noqa: E402
from ml.model import stability  # noqa: E402
from ml.model.train import (COUNT_OF, grades, prepare,  # noqa: E402
                            relevance_problem)

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"
OUT = DATA / "item3_relevance.csv"
LABEL = "label_alias"

# The cut dates of §23.5, where §24 measured the model that ships.
DATES = ["2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
         "2026-03-02", "2026-04-02", "2026-05-04"]
# §24.2, label_alias: its lift over popularity at those dates. The binary
# sweep below has to give exactly these.
SHIPPED = [4.42, 4.86, 5.70, 4.02, 3.01, 4.08, 2.30]

MARGIN = 0.25
NEED = 6  # dates of the 7 where graded nDCG@20 must be higher


def verdict(b: pd.DataFrame, g: pd.DataFrame) -> tuple[bool | None,
                                                       list[str]]:
    """§25.3 applied to the two sweeps, one row per date in each. Returns
    (does graded ship, the reasoning line by line); None when the rule
    cannot be applied because a date was not measured."""
    both = ~b["skipped"].astype(bool).to_numpy() \
        & ~g["skipped"].astype(bool).to_numpy()
    n = int(both.sum())
    if n != len(DATES):
        return None, [f"Only {n} of the {len(DATES)} dates were measured "
                      f"by both. The rule was written for all "
                      f"{len(DATES)}, so there is no verdict."]
    b, g = b[both], g[both]
    worst_b = float(b["lift_vs_pop"].astype(float).min())
    worst_g = float(g["lift_vs_pop"].astype(float).min())
    beats = int(g["beats_pop"].astype(bool).sum())
    floor, bar = round(worst_b - MARGIN, 2), round(worst_b + MARGIN, 2)
    # Lifts are stored to 2 places, so compare at 2 places: exactly 0.25x
    # below still holds, exactly 0.25x above is not yet "more than".
    holds_lift = worst_g >= floor - 1e-9
    holds_pop = beats == n
    wins_lift = worst_g > bar + 1e-9
    better = int((g["ndcg_20_graded"].astype(float).to_numpy()
                  > b["ndcg_20_graded"].astype(float).to_numpy()).sum())
    wins_ndcg = better >= NEED
    ships = holds_lift and holds_pop and (wins_lift or wins_ndcg)

    def line(text: str, ok: bool, value: str) -> str:
        return f"     {text:<54}{'yes' if ok else 'no'}, {value}"

    lines = [
        "1. graded loses nothing that matters",
        line(f"worst lift at least {floor:.2f}x "
             f"(binary's {worst_b:.2f}x - {MARGIN}x)", holds_lift,
             f"{worst_g:.2f}x"),
        line("beats popularity at every date", holds_pop, f"{beats}/{n}"),
        "2. graded wins clearly somewhere",
        line(f"worst lift above {bar:.2f}x "
             f"(binary's {worst_b:.2f}x + {MARGIN}x)", wins_lift,
             f"{worst_g:.2f}x"),
        line(f"graded nDCG@20 higher at {NEED}+ of {n} dates", wins_ndcg,
             f"{better}/{n}"),
    ]
    return ships, lines


def main() -> None:
    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found. Run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    assert_no_holdout(df, "item3_relevance.py")
    problem = relevance_problem(df, LABEL, "lambdarank", "graded")
    if problem:
        sys.exit(f"item3_relevance.py: {problem}")
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    # What graded training will be told, described once.
    g = grades(df, LABEL)
    pos = g[g > 0]
    print(f"\n{len(df):,} dev rows, {len(pos):,} positive under {LABEL}, "
          f"graded by {COUNT_OF[LABEL]}:")
    for k, name in zip([1, 2, 3, 4], ["1 package", "2 to 6", "7 to 19",
                                      "20 or more"]):
        print(f"  grade {k}  {name:<11} {int((pos == k).sum()):>6,}  "
              f"{(pos == k).mean():>6.1%}")

    runs = {}
    for rel in ("binary", "graded"):
        print(f"\n{rel}: fitting at {len(DATES)} cut dates ...", flush=True)
        t = stability.run_label(df, LABEL, feats, "lambdarank", "cv",
                                DATES, relevance=rel)
        runs[rel] = t.set_index("cut").reindex(DATES)
    b, gr = runs["binary"], runs["graded"]

    print(f"\n{LABEL.upper()} AT THE SEVEN DATES OF §23.5, BINARY AGAINST "
          "GRADED\n")
    got = [None if s else round(float(v), 2)
           for s, v in zip(b["skipped"].astype(bool), b["lift_vs_pop"])]
    reproduced = got == SHIPPED
    if reproduced:
        print("  self-check: binary reproduces §24.2, lift for lift")
    else:
        print("  self-check: binary DOES NOT reproduce §24.2")
        print(f"      §24.2    {SHIPPED}\n      this run {got}")

    def cell(t: pd.DataFrame, d: str, col: str, fmt: str) -> str:
        v = t.loc[d, col]
        return "-" if bool(t.loc[d, "skipped"]) or pd.isna(v) else fmt % v

    print(f"\n  {'':<12}{'lift over popularity':^22}"
          f"{'nDCG@20, graded gains':^32}{'trees':^17}".rstrip())
    print(f"  {'cut':<12}{'binary':>12}{'graded':>10}"
          f"{'binary':>12}{'graded':>10}{'change':>10}"
          f"{'binary':>9}{'graded':>8}")
    for d in DATES:
        change = "-"
        if not (bool(b.loc[d, "skipped"]) or bool(gr.loc[d, "skipped"])):
            change = "%+.4f" % (float(gr.loc[d, "ndcg_20_graded"])
                                - float(b.loc[d, "ndcg_20_graded"]))
        print(f"  {d:<12}{cell(b, d, 'lift_vs_pop', '%.2fx'):>12}"
              f"{cell(gr, d, 'lift_vs_pop', '%.2fx'):>10}"
              f"{cell(b, d, 'ndcg_20_graded', '%.4f'):>12}"
              f"{cell(gr, d, 'ndcg_20_graded', '%.4f'):>10}"
              f"{change:>10}"
              f"{cell(b, d, 'trees', '%d'):>9}"
              f"{cell(gr, d, 'trees', '%d'):>8}")

    def stat(t: pd.DataFrame, col: str, how: str) -> float:
        v = t.loc[~t["skipped"].astype(bool), col].astype(float)
        return float(getattr(v, how)())

    print(f"  {'worst':<12}{stat(b, 'lift_vs_pop', 'min'):>11.2f}x"
          f"{stat(gr, 'lift_vs_pop', 'min'):>9.2f}x")
    print(f"  {'median':<12}{stat(b, 'lift_vs_pop', 'median'):>11.2f}x"
          f"{stat(gr, 'lift_vs_pop', 'median'):>9.2f}x"
          f"{stat(b, 'ndcg_20_graded', 'median'):>12.4f}"
          f"{stat(gr, 'ndcg_20_graded', 'median'):>10.4f}")

    print("\n  The other numbers, for the record (medians over the dates):")
    print(f"  {'':<12}{'PR-AUC':>10}{'precision@10':>15}{'nDCG@20':>10}")
    for name, t in (("binary", b), ("graded", gr)):
        print(f"  {name:<12}{stat(t, 'pr_auc', 'median'):>10.4f}"
              f"{stat(t, 'p_at_10', 'median'):>15.4f}"
              f"{stat(t, 'ndcg_20', 'median'):>10.4f}")

    print("\nTHE RULE, NOTES §25.3, fixed before this run\n")
    if not reproduced:
        print("  No verdict. The binary sweep is not the model §24 measured,")
        print("  so the comparison is not the one the rule was written for.")
        print("  Find out why before reading anything above.")
    else:
        ships, lines = verdict(b, gr)
        for line in lines:
            print(f"  {line}")
        if ships is None:
            pass
        elif ships:
            print("\n  VERDICT: graded ships. Its model replaces the binary "
                  "one (NOTES §25.5).")
        else:
            print("\n  VERDICT: binary stays. Graded is kept as an option "
                  "and not shipped (NOTES §25.5).")

    both = pd.concat([b.reset_index(), gr.reset_index()], ignore_index=True)
    both.to_csv(OUT, index=False)
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
