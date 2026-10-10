"""
Did the model get better, or did the data get honest?

    python scripts/fold_effect.py --cut 2026-06-13
    python scripts/fold_effect.py --cut 2026-06-13 --label label_alias

--cut is required and must fall BEFORE HOLDOUT_START, where the frozen
holdout begins (ml/holdout.py). §11.5 was measured on 15 Sep at
2026-08-15, a date that now sits inside the holdout, so that exact run can
no longer be repeated. The table in §11.5 stays the record of it. Holdout
rows are dropped from both datasets before anything is split.

THE QUESTION, asked by Varad on 15 Sep and worth asking.

PR-AUC went 0.3465 -> 0.5137 in the same step that removed 13,694
inherited-duplicate rows (NOTES §10.2). Those two facts arriving together
is exactly the shape of a number nobody should take at face value. One
version pair was 30% of the old dataset, so the old model was partly
being scored on how well it handled transformers.

"We checked" beats a shrug, and this is the check.

HOW IT IS MADE FAIR.

Everything is held still except the duplicates:

  same label, same features, same objective, same CV stopping rule,
  same code — and crucially, THE SAME CUT DATE.

That last one is the part that is easy to get wrong. The normal split
cuts at a QUANTILE of the rows, so the two datasets — 32,405 rows and
19,121 — would cut at different dates and be scored on different test
sets. Comparing those would measure the split as much as the fold. A
fixed calendar date gives both runs a test half drawn from the same
window; the amplified one simply has thousands of duplicate rows inside
it.

WHAT THE ANSWER LOOKS LIKE.

The floor moves. It has to: removing 13,694 guaranteed negatives raises
the positive rate, and PR-AUC's floor IS the positive rate. So raw PR-AUC
is not the comparison — VS FLOOR is, and so is lift over popularity,
because both are computed within a dataset.

  If vs-floor is roughly the same on both, the fold did not make the
  model better. It made the NUMBER bigger by removing negatives, and the
  honest sentence is "the dataset was wrong, the model was always this
  good, and the old score was depressed by duplicates."

  If vs-floor improves, the duplicates were actively harming the ranker
  — which is plausible for lambdarank, where thousands of identical rows
  inside one group distort the group structure being optimised.

Both are publishable. Only one of them is "the model improved", and
without this script we would not know which.

Nothing is written. No file in data/ or artifacts/ is touched — this runs
the pipeline in memory so it cannot clobber the shipped model.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import (BOOLEAN, CATEGORICAL, NUMERIC,  # noqa: E402
                               add_features)
from ml.features.labels import add_labels  # noqa: E402
from ml.holdout import HOLDOUT_START, drop_holdout  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import evaluate, n_rankable  # noqa: E402
from ml.model.train import GROUP, fit_cv, prepare, score_with  # noqa: E402

DATA = pathlib.Path("data")
USAGE = DATA / "usage.csv"

# Default pair: the pre-fold file kept deliberately for this comparison,
# and the current one.
BEFORE = DATA / "changes-amplified.csv"
AFTER = DATA / "changes.csv"


def build(changes_path: pathlib.Path, usage: pd.DataFrame, cut: pd.Timestamp,
          label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """changes.csv -> (train, test), split at a FIXED DATE."""
    changes = pd.read_csv(changes_path)
    df = add_features(add_labels(changes, usage))
    # changes.csv holds everything, the frozen holdout included, so it is
    # dropped here rather than asserted absent. Same order as build.py:
    # features on the full frame, then the holdout comes off.
    df = drop_holdout(df)

    when = pd.to_datetime(df["released_at"], errors="coerce")
    # Undated rows train and never test — the same rule build.temporal_split
    # uses. Copying the rule rather than the function, because that one
    # cuts on a quantile and this must cut on a date.
    is_test = (when > cut).fillna(False)
    df = prepare(df)
    return (df[~is_test].sort_values(GROUP), df[is_test].sort_values(GROUP))


def run(name: str, changes_path: pathlib.Path, usage: pd.DataFrame,
        cut: pd.Timestamp, label: str, objective: str) -> dict:
    train, test = build(changes_path, usage, cut, label)
    if test.empty or not test[label].sum():
        sys.exit(f"{name}: no {label} positives between {cut.date()} and "
                 f"{HOLDOUT_START.date()}, where the holdout begins.\n"
                 "Nothing to score. Pick an earlier --cut.")
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    model, trees, _folds = fit_cv(train, feats, label, objective)
    test = test.copy()
    test["model"] = score_with(model, test, feats)
    test = add_baseline_scores(train, test, label)

    m = evaluate(test, "model", label)
    pop = evaluate(test, "popularity", label)
    floor = float(test[label].mean())

    r = {
        "name": name,
        "rows": len(train) + len(test),
        "train": len(train),
        "test": len(test),
        "pos": int(test[label].sum()),
        "floor": floor,
        "trees": trees,
        "rankable": n_rankable(test, label, 10),
        "pr_auc": m["pr_auc"],
        "pop": pop["pr_auc"],
        "vs_floor": m["pr_auc"] / floor if floor else float("nan"),
        "lift": m["pr_auc"] / pop["pr_auc"] if pop["pr_auc"] else float("nan"),
        "p10": m["precision_at_10"],
        "ndcg": m["ndcg_at_20"],
    }
    print(f"  {name:<22} {r['rows']:>7,} rows -> test {r['test']:>6,}"
          f"   {r['pos']:>4} positive ({floor:.2%})   {trees} trees")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Separate 'the model improved' from 'the data got honest'.")
    ap.add_argument("--before", default=str(BEFORE))
    ap.add_argument("--after", default=str(AFTER))
    ap.add_argument("--cut", required=True,
                    help="fixed split date, applied to BOTH datasets. Must "
                         f"be before {HOLDOUT_START.date()}, where the "
                         "frozen holdout begins.")
    ap.add_argument("--label", default="label_alias",
                    choices=["label", "label_scoped", "label_alias"])
    ap.add_argument("--objective", default="lambdarank")
    args = ap.parse_args()

    before, after = pathlib.Path(args.before), pathlib.Path(args.after)
    for p in (before, after, USAGE):
        if not p.exists():
            sys.exit(f"{p} not found.")

    cut = pd.Timestamp(args.cut)
    if cut >= HOLDOUT_START:
        sys.exit(f"--cut {cut.date()} is inside the frozen holdout (from "
                 f"{HOLDOUT_START.date()}). With holdout rows removed, the "
                 "test half\nwould be empty. Pick a date before it; "
                 "ml/holdout.py explains the freeze.")
    usage = pd.read_csv(USAGE)

    print(f"\nlabel {args.label}   cut held fixed at {cut.date()} for both\n")
    a = run("BEFORE the fold", before, usage, cut, args.label, args.objective)
    b = run("AFTER the fold", after, usage, cut, args.label, args.objective)

    print("\n" + "=" * 72)
    print("  SAME CODE, SAME CUT DATE, SAME LABEL — ONLY THE DUPLICATES DIFFER")
    print("=" * 72)
    rows = [
        ("rows in dataset", f"{a['rows']:,}", f"{b['rows']:,}"),
        ("test positives", f"{a['pos']}", f"{b['pos']}"),
        ("floor (positive rate)", f"{a['floor']:.4f}", f"{b['floor']:.4f}"),
        ("PR-AUC", f"{a['pr_auc']:.4f}", f"{b['pr_auc']:.4f}"),
        ("vs floor", f"{a['vs_floor']:.1f}x", f"{b['vs_floor']:.1f}x"),
        ("popularity PR-AUC", f"{a['pop']:.4f}", f"{b['pop']:.4f}"),
        ("lift over popularity", f"{a['lift']:.2f}x", f"{b['lift']:.2f}x"),
        ("precision@10", f"{a['p10']:.4f}", f"{b['p10']:.4f}"),
        ("nDCG@20", f"{a['ndcg']:.4f}", f"{b['ndcg']:.4f}"),
        ("rankable pairs", f"{a['rankable']}", f"{b['rankable']}"),
        ("trees (CV median)", f"{a['trees']}", f"{b['trees']}"),
    ]
    print(f"  {'':<24}{'before':>12}{'after':>12}")
    for lbl, x, y in rows:
        print(f"  {lbl:<24}{x:>12}{y:>12}")

    print("\n" + "-" * 72)
    d_floor = b["floor"] / a["floor"] if a["floor"] else float("nan")
    d_vs = b["vs_floor"] / a["vs_floor"] if a["vs_floor"] else float("nan")
    d_lift = b["lift"] / a["lift"] if a["lift"] else float("nan")
    print(f"  the floor rose {d_floor:.2f}x — removing guaranteed negatives "
          "does that,")
    print("  and it raises PR-AUC on its own without the model changing.")
    print(f"  vs-floor moved {d_vs:.2f}x   ·   lift over popularity moved "
          f"{d_lift:.2f}x")

    print()
    if d_vs < 1.15 and d_lift < 1.15:
        print("  READ IT AS: the data got honest, not the model better.")
        print("  Normalised for the floor, the ranker is doing the same job it")
        print("  always was. The old score was DEPRESSED by 13,694 duplicate")
        print("  negatives; removing them raised the number without changing")
        print("  the skill. Say that, rather than claiming an improvement.")
    elif d_vs >= 1.15 or d_lift >= 1.15:
        print("  READ IT AS: both. The floor moved AND the ranker improved")
        print("  relative to it. Plausible for lambdarank specifically —")
        print("  thousands of identical rows inside one group distort the")
        print("  group structure the objective optimises, so removing them")
        print("  is not merely cosmetic. Quote both numbers, not just PR-AUC.")

    print("\n  Caveat that applies either way: these two runs share most of")
    print("  their training data, so this is a sensitivity check, not a")
    print("  significance test. There is no p-value here and there should")
    print("  not be one.")


if __name__ == "__main__":
    main()
