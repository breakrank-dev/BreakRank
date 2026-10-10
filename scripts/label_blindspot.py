"""
Does the model rank, or does it only separate what the label can see?

    python scripts/label_blindspot.py
    python scripts/label_blindspot.py --label label_scoped

WHY THIS EXISTS.

Measured 15 Sep, on 19,121 labelled rows:

    method on a class   12,373 rows    1.12% positive
    module-level         6,748 rows   12.34% positive

Eleven times. That gap is not a fact about Python. It is a fact about our
usage index, which is built from IMPORT STATEMENTS. People write
`from pandas import read_csv`; nobody writes `from pandas import
DataFrame.append` — they call it on an object. So a module-level symbol
can be seen by the scanner and a method essentially cannot.

`DataFrame.append` being removed broke thousands of codebases. Our label
scores changes like it near zero. 65% of the dataset is methods.

THE WORRY THIS SCRIPT TESTS. `public_depth` is the model's top feature at
35.6% of gain. Module-level symbols have low public_depth; methods have
higher. So public_depth partly encodes "is this the KIND of symbol our
label can observe at all" — which would make the headline feature a proxy
for our own blind spot. That is the same class of error as NOTES §5.3,
in a form §5.3 did not cover.

THE TEST. Split the test set by symbol shape and score each half on its
own. Lift over popularity is the comparable number — PR-AUC is not, since
the two halves have wildly different positive rates and PR-AUC's floor IS
the positive rate.

  If the model beats popularity INSIDE the method subset, it is ordering
  changes rather than sorting symbol kinds, and the finding survives with
  a stated limitation.

  If it collapses to ~1.0x inside methods, then most of its measured skill
  is "module-level symbols matter more" — true, partly circular, and a
  very different claim from the one currently in §1.

Either answer is worth having. The second one would be worth having
before an examiner has it.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import assert_no_holdout  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import evaluate, n_rankable  # noqa: E402
from ml.model.train import GROUP, fit_cv, prepare, score_with  # noqa: E402

FEATURES = pathlib.Path("data") / "features.csv"


def is_method(symbol: pd.Series) -> pd.Series:
    """Is the leaf attached to a class rather than a module?

    Heuristic, and a deliberately crude one: the component BEFORE the leaf
    starts with an uppercase letter. `pandas.DataFrame.append` yes;
    `pandas.read_csv` no. PEP 8 makes this right far more often than not,
    and the alternative — asking griffe for each symbol's parent kind —
    would mean re-parsing 415 packages to answer a question about the
    shape of the label.

    It will misfile a module named `HTTPStatus` and a class named `utils`.
    Both are rare enough not to move a 12,000-row aggregate.
    """
    parts = symbol.str.split(".")
    return parts.apply(lambda p: len(p) >= 2 and p[-2][:1].isupper())


# The same gate stability.py applies (§11.2). A top-10 metric over three
# version pairs is not a weak measurement of ranking quality, and this
# script quoted one before the gate was copied across — precision@10
# 0.2667 over 3 pairs, in a project that had just added the rule.
MIN_RANKABLE_PAIRS = 10


def report(name: str, test: pd.DataFrame, label: str) -> dict | None:
    pos = int(test[label].sum())
    rankable = n_rankable(test, label, 10)
    floor = float(test[label].mean())
    print(f"\n  {name}")
    print(f"    {len(test):,} rows   {pos} positive ({floor:.2%})"
          f"   {rankable} rankable pairs")
    if pos < 10:
        print("    too few positives to score — not reported rather than "
              "reported badly")
        return None

    m = evaluate(test, "model", label)
    pop = evaluate(test, "popularity", label)

    # PR-AUC is computed across every row, so it survives here. The floor
    # is the positive rate, and vs-floor is the one ratio that means the
    # same thing in both subsets.
    vs_floor = m["pr_auc"] / floor if floor else float("nan")
    print(f"    model PR-AUC {m['pr_auc']:.4f}   floor {floor:.4f}"
          f"   {vs_floor:.1f}x the floor")

    # LIFT OVER POPULARITY IS ONLY MEANINGFUL WHILE POPULARITY BEATS
    # CHANCE. Measured 15 Sep: inside the method subset popularity scores
    # 0.0094 against a 0.0120 floor — it is WORSE than random there, so
    # dividing by it manufactures a huge number (32x) out of an unstable
    # near-zero denominator. That is not 32x of skill, it is a broken
    # ratio, and it would have gone straight into the report.
    if pop["pr_auc"] > floor:
        lift = m["pr_auc"] / pop["pr_auc"]
        print(f"    popularity {pop['pr_auc']:.4f}   lift {lift:.2f}x")
    else:
        lift = float("nan")
        print(f"    popularity {pop['pr_auc']:.4f}  <- BELOW the floor: the")
        print("      baseline is worse than chance in this subset, so lift")
        print("      over it is not a meaningful number. Use vs-floor.")

    if rankable >= MIN_RANKABLE_PAIRS:
        print(f"    precision@10 {m['precision_at_10']:.4f}   "
              f"nDCG@20 {m['ndcg_at_20']:.4f}")
    else:
        print(f"    precision@10 / nDCG not reported — {rankable} rankable "
              f"pairs, min {MIN_RANKABLE_PAIRS}")

    return {"name": name, "rows": len(test), "pos": pos, "floor": floor,
            "pr_auc": m["pr_auc"], "pop": pop["pr_auc"], "lift": lift,
            "vs_floor": vs_floor, "rankable": rankable}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Does the model rank inside the label's blind spot?")
    ap.add_argument("--label", default="label_alias",
                    choices=["label", "label_scoped", "label_alias"])
    ap.add_argument("--objective", default="lambdarank")
    args = ap.parse_args()
    label = args.label

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    assert_no_holdout(df, "label_blindspot.py")
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    full_train = df[df.split == "train"].sort_values(GROUP)
    test = df[df.split == "test"].sort_values(GROUP)

    # ONE model, trained on everything, exactly as shipped. The split
    # happens at SCORING time only — training a separate model per subset
    # would answer a different question ("could a specialist do it?")
    # rather than this one ("what is the shipped model actually doing?").
    model, trees, _folds = fit_cv(full_train, feats, label, args.objective)
    test = test.copy()
    test["model"] = score_with(model, test, feats)
    test = add_baseline_scores(full_train, test, label)

    meth = is_method(test["symbol"])
    print(f"\nlabel {label}   {trees} trees   test {len(test):,} rows")
    print("=" * 66)
    print("  Read vs-floor, not raw PR-AUC and not lift. PR-AUC's floor is")
    print("  the positive rate, and that is what differs most between these")
    print("  groups; popularity is worse than chance in one of them, so lift")
    print("  over it is undefined there.")
    print()
    print("  AND DO NOT COMPARE vs-floor ACROSS THE TWO GROUPS EITHER. A")
    print("  rarer positive class produces a larger ratio for the same real")
    print("  skill, so 'methods 25x beats module-level 7x' is NOT a claim")
    print("  this table supports. Each number says only: within this group,")
    print("  the model is far above chance.")

    rows = [report("EVERYTHING", test, label),
            report("MODULE-LEVEL symbols — what the label can see", test[~meth], label),
            report("METHODS ON CLASSES — the blind spot", test[meth], label)]
    rows = [r for r in rows if r]

    print("\n" + "=" * 66)
    print("  WHAT THIS MEANS")
    print("=" * 66)
    by = {r["name"].split(" —")[0]: r for r in rows}
    m, mod = by.get("METHODS ON CLASSES"), by.get("MODULE-LEVEL symbols")
    if not m or not mod:
        print("  One group could not be scored; no conclusion drawn.")
        return

    print(f"  positive rate   module-level {mod['floor']:.2%}   "
          f"methods {m['floor']:.2%}   "
          f"({mod['floor'] / max(m['floor'], 1e-9):.1f}x)")
    print(f"  vs floor        module-level {mod['vs_floor']:>5.1f}x   "
          f"methods {m['vs_floor']:>5.1f}x")
    print("  ^ this is the comparison to quote. It means the same thing in")
    print("    both subsets; lift over popularity does not, because in one")
    print("    of them popularity is worse than chance.")

    # Judged on vs-floor, not on lift. PR-AUC across thousands of rows is a
    # real measurement even where the per-pair metrics are not.
    if m["vs_floor"] >= 5:
        print("\n  The model still ranks INSIDE the blind spot. It is ordering")
        print("  changes, not sorting symbol kinds — so the result survives,")
        print("  and the under-labelling of methods is a limitation to state")
        print("  rather than a hole underneath the headline.")
    elif m["vs_floor"] >= 2:
        print("\n  Weak but present inside the blind spot. Report both numbers;")
        print("  the overall result is carried more by module-level rows than")
        print("  the single figure in §1 suggests.")
    else:
        print("\n  ** The model does NOT rank inside the method subset.")
        print("  ** Most of its measured skill is separating module-level")
        print("  ** symbols from methods — which is partly a real property of")
        print("  ** the ecosystem and partly our scanner's blind spot. The")
        print("  ** claim in §1 needs restating before it is quoted again.")

    if m["rankable"] < MIN_RANKABLE_PAIRS:
        print(f"\n  Caveat to carry with the above: the method subset has only")
        print(f"  {m['rankable']} rankable version pairs, so nothing about "
              "TOP-10 quality")
        print("  inside it has been measured. PR-AUC across 3,341 rows is a")
        print("  real number; precision@10 over 3 pairs would not have been.")

    if pd.notna(mod["lift"]) and pd.isna(m["lift"]):
        print("\n  Worth its own line: popularity is WORSE THAN CHANCE among")
        print("  methods. Big packages have enormous numbers of methods and")
        print("  almost no labelled ones, so sorting by download count")
        print("  actively misleads exactly where the label is weakest.")

    print("\n  Either way: the usage index sees imports, not attribute access.")
    print("  Fixing that is a Track B change (record `obj.method()` call sites,")
    print("  not just import statements), not a model change.")


if __name__ == "__main__":
    main()
