"""
Is the result real, or is it one feature wearing a hat?

    python ml/model/ablate.py
    python ml/model/ablate.py --label label_scoped

The first ranker scored 1.85x popularity, and its top two features were
module_depth and name_length — not `kind`, not `package_rank`. Before
anyone celebrates that, there is a specific reason to be suspicious.

THE WORRY. The label is "does this exact symbol path appear in the usage
index". The usage index records paths AS DOWNSTREAM CODE WRITES THEM, and
downstream code writes short ones: `from pandas import read_csv` becomes
pandas.read_csv, depth 1. griffe reports the change at the definition,
pandas.io.parsers.readers.read_csv, depth 4 — which the strict join then
scores zero.

So shallow symbols look important partly because deep ones are where our
JOIN FAILS. module_depth may be predicting our own measurement error.
That is still a real pattern (top-level symbols genuinely are imported
more), but the size of the effect would be inflated, and a model resting
on it is resting on the re-export gap.

This script refits the model with features removed and reports what each
group was worth. Three questions it answers:

  drop path shape   (module_depth, name_length, is_top_level)
      If PR-AUC collapses, the model IS the depth heuristic. Say so.

  drop popularity   (package_rank, package_churn, release_size)
      If PR-AUC collapses, the model is the popularity baseline with
      extra steps, and popularity is the thing it claims to beat.

  keep only kind    the per-change signal that cannot be either of those

Run it on BOTH labels. If path shape matters much less under
label_scoped — where deep re-exported symbols are correctly positive —
that is direct evidence the effect was measurement, not ecosystem.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.model.metrics import evaluate  # noqa: E402
from ml.model.train import (fit_model, prepare, score_with,  # noqa: E402
                            split_valid, GROUP)

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"

PATH_SHAPE = ["module_depth", "name_length", "is_top_level"]
POPULARITY = ["package_rank", "package_churn", "release_size"]
PER_CHANGE = ["kind", "bump", "is_private", "is_dunder", "in_dunder_all",
              "is_version_string", "has_sub_target"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature ablations.")
    ap.add_argument("--label", default="label",
                    choices=["label", "label_scoped"])
    ap.add_argument("--objective", default="lambdarank")
    args = ap.parse_args()
    label = args.label

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    everything = NUMERIC + BOOLEAN + CATEGORICAL

    full_train = df[df.split == "train"].sort_values(GROUP)
    test = df[df.split == "test"].sort_values(GROUP)
    train, valid, _ = split_valid(full_train)
    train, valid = train.sort_values(GROUP), valid.sort_values(GROUP)

    runs = {
        "everything": everything,
        "no path shape": [f for f in everything if f not in PATH_SHAPE],
        "no popularity": [f for f in everything if f not in POPULARITY],
        "path shape only": PATH_SHAPE,
        "popularity only": POPULARITY,
        "per-change only": PER_CHANGE,
    }

    print(f"\nlabel {label}   test {len(test):,} rows "
          f"({test[label].mean():.2%} positive)\n")

    out = {}
    for name, feats in runs.items():
        model = fit_model(train, valid, feats, label, args.objective)
        scored = test.copy()
        scored["s"] = score_with(model, test, feats)
        m = evaluate(scored, "s", label)
        m["trees"] = getattr(model, "best_iteration_", None) or 600
        m["n_features"] = len(feats)
        out[name] = m

    t = pd.DataFrame(out).T[["n_features", "trees", "pr_auc",
                             "precision_at_10", "ndcg_at_20"]]
    base = t.loc["everything", "pr_auc"]
    t["vs_full"] = (t["pr_auc"] / base).round(2)
    print(t.round(4).to_string())

    print(f"\nfull model PR-AUC {base:.4f}")
    for group, name in ((PATH_SHAPE, "no path shape"),
                        (POPULARITY, "no popularity")):
        lost = 1 - out[name]["pr_auc"] / base
        label_txt = name.replace("no ", "")
        print(f"  removing {label_txt:<12} costs {lost:>6.1%} of PR-AUC")

    solo = max(("path shape only", "popularity only", "per-change only"),
               key=lambda k: out[k]["pr_auc"])
    if out[solo]["pr_auc"] > 0.85 * base:
        print(f"\n** '{solo}' alone reaches {out[solo]['pr_auc'] / base:.0%} "
              f"of the full model.\n** The other features are close to "
              "decoration. Report the honest\n** version: this is largely a "
              f"{solo} heuristic.")
    else:
        print(f"\nNo single group reaches 85% of the full model "
              f"(best is '{solo}' at {out[solo]['pr_auc'] / base:.0%}),\n"
              "so the result rests on a combination rather than one "
              "dressed-up feature.")

    out_path = DATA / f"ablation_{label}.csv"
    t.to_csv(out_path)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
