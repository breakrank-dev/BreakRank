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
from ml.model.train import (cv_tree_count, fit_fixed, prepare,  # noqa: E402
                            score_with, GROUP)

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"

# PATH_SHAPE stays at the ORIGINAL THREE. public_depth and has_export_path
# were briefly folded in here, which silently broke the one comparison this
# script exists for: NOTES §5.3 reports "path shape alone = 105% under the
# strict label, 47% under scoped", and a group that grew from three features
# to five is not the same group. The 97% it then produced under label_alias
# looked like a finding and was partly just a bigger bucket.
PATH_SHAPE = ["module_depth", "name_length", "is_top_level"]

# The alias graph's two features, kept SEPARATE on purpose, because they
# answer a different question. module_depth says where a symbol is DEFINED.
# public_depth says how far you have to reach to get at it — the shortest
# name anyone can import. A model leaning on the first is using a proxy for
# obscurity; a model leaning on the second is using reachability, which is
# much closer to the thing we actually claim to predict.
#
# Worth watching for circularity, and saying so out loud: the usage index
# records paths as downstream code writes them, and downstream code writes
# short ones. So a symbol with a short public path has more ways to match
# and matches on the kind of path the index is full of. That is a real
# property of the ecosystem AND a property of how the label is built, and
# this split is what lets the two be argued about with numbers.
REACHABILITY = ["public_depth", "has_export_path"]

# Added 12 Sep 2026, and kept on its own for the same reason REACHABILITY
# is: it is a new IDEA, not a refinement of an old one. Every other feature
# here describes the symbol; this one describes how far the change spreads
# inside its own library. It arrived as a side effect of folding
# inherited-member repeats (NOTES §10.2), so the honest thing is to test
# whether it earns a slot rather than assume the number is free signal.
BLAST_RADIUS = ["inherited_by"]

# Added 16 Sep 2026. The two features that need MORE THAN ONE RELEASE to
# exist — everything else in the table can be read off a single diff.
#
# Kept together and kept separate from everything else because the claim
# being tested is about the idea, not the columns: does knowing what the
# library did BEFORE this release help predict what this release breaks?
# The project book expects was_deprecated_before to be the strongest
# single feature in the set, and an expectation that specific is worth
# giving its own row rather than burying inside "per-change".
#
# Read the "history only" row against POPULARITY's, not against
# everything's. Popularity is the baseline the whole project exists to
# beat; if two features computed from the version chain alone land near
# it, that is the finding.
HISTORY = ["was_deprecated_before", "prior_breaks_in_module"]

POPULARITY = ["package_rank", "package_churn", "release_size"]
PER_CHANGE = ["kind", "bump", "is_private", "is_dunder", "in_dunder_all",
              "is_version_string", "has_sub_target"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature ablations.")
    ap.add_argument("--label", default="label",
                    choices=["label", "label_scoped", "label_alias"])
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--trees", type=int, default=None,
                    help="fix the tree count for EVERY run. Default: the "
                         "CV-chosen count for the full feature set.")
    args = ap.parse_args()
    label = args.label

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    everything = NUMERIC + BOOLEAN + CATEGORICAL

    full_train = df[df.split == "train"].sort_values(GROUP)
    test = df[df.split == "test"].sort_values(GROUP)

    drop = lambda g: [f for f in everything if f not in g]  # noqa: E731
    runs = {
        "everything": everything,
        "no path shape": drop(PATH_SHAPE),
        "no reachability": drop(REACHABILITY),
        "no path+reach": drop(PATH_SHAPE + REACHABILITY),
        "no blast radius": drop(BLAST_RADIUS),
        "no history": drop(HISTORY),
        "no popularity": drop(POPULARITY),
        "path shape only": PATH_SHAPE,
        "reachability only": REACHABILITY,
        "history only": HISTORY,
        "popularity only": POPULARITY,
        "per-change only": PER_CHANGE,
    }

    # EVERY RUN GETS THE SAME NUMBER OF TREES, and this is the whole
    # comparison. Until 14 Sep each subset chose its own count by early
    # stopping on a holdout slice, and the counts came back 4, 10, 11, 27,
    # 40, 47, 53, 59 — a fifteen-fold range. "Removing blast radius costs
    # 25% of PR-AUC" was then indistinguishable from "that run happened to
    # get 4 trees", because a 4-tree model is barely a model.
    #
    # It was also the holdout stopping rule that §5.7 already replaced
    # everywhere else: the validation slice is 2x denser in positives than
    # test, so it stops at the wrong place, and it was still in here.
    #
    # An ablation is a controlled experiment. The thing being varied is the
    # FEATURE SET, so model size has to be held still — otherwise the table
    # measures two things at once and reports it as one number.
    # cv_tree_count returns (median, per-fold counts) — the folds are worth
    # printing, not discarding, because their spread is the reason this
    # whole fixed-count change exists.
    if args.trees:
        n_trees, folds = args.trees, []
    else:
        n_trees, folds = cv_tree_count(full_train, everything, label,
                                       args.objective)

    print(f"\nlabel {label}   test {len(test):,} rows "
          f"({test[label].mean():.2%} positive)")
    print(f"every run fixed at {n_trees} trees"
          + (f" (CV median of {folds})" if folds else " (set by --trees)"))
    print("Fixed on purpose: letting each subset pick its own size makes "
          "the\ncolumns incomparable. See the note in the source.\n")

    # ----------------------------------------------------------------
    # CONTROLS. An ablation number means nothing without knowing how big
    # a number this table produces when NOTHING is really removed.
    #
    # Measured 14 Sep: `inherited_by` had exactly zero gain — the model
    # was offered it and never split on it — yet dropping it moved PR-AUC
    # by 12.6%. Dropping a whole feature changes LightGBM's binning and
    # column sampling, so the fit moves even when the feature was unused.
    # Meanwhile "removing path shape costs 10.4%" was being read as a
    # result. It was smaller than the noise.
    #
    # So: fit the full model, ask it which features it never split on, and
    # drop each of those as a control run. Whatever those cost IS the
    # floor, measured on this dataset with this tree count rather than
    # guessed. Everything below it reads as nothing.
    #
    # Discovered, not hardcoded — which features go unused changes with
    # the label and the data, and a stale list would be worse than none.
    full_model = fit_fixed(full_train, everything, label, n_trees,
                           args.objective)
    gains = dict(zip(everything,
                     full_model.booster_.feature_importance("gain")))
    unused = [f for f in everything if gains.get(f, 0) <= 0]
    for f in unused[:3]:
        runs[f"control: drop {f}"] = drop([f])
    if unused:
        print(f"control runs added for {len(unused[:3])} feature(s) at zero "
              f"gain: {', '.join(unused[:3])}")
        print("Their cost is this table's noise floor, not a finding.\n")
    else:
        print("no zero-gain features — no control available, so treat "
              "small\ndifferences below with corresponding suspicion.\n")

    out = {}
    for name, feats in runs.items():
        model = fit_fixed(full_train, feats, label, n_trees, args.objective)
        scored = test.copy()
        scored["s"] = score_with(model, test, feats)
        m = evaluate(scored, "s", label)
        m["trees"] = n_trees
        m["n_features"] = len(feats)
        out[name] = m

    t = pd.DataFrame(out).T[["n_features", "trees", "pr_auc",
                             "precision_at_10", "ndcg_at_20"]]
    base = t.loc["everything", "pr_auc"]
    t["vs_full"] = (t["pr_auc"] / base).round(2)
    print(t.round(4).to_string())

    # The floor: the largest swing produced by removing a feature the model
    # never used. Absolute value — a control can move PR-AUC UP as easily
    # as down, and either direction is the same noise.
    controls = [n for n in out if n.startswith("control: drop ")]
    floor = max((abs(1 - out[n]["pr_auc"] / base) for n in controls),
                default=0.0)

    print(f"\nfull model PR-AUC {base:.4f}")
    if controls:
        print(f"NOISE FLOOR {floor:>6.1%}  <- removing a feature the model "
              "never split on")
        print("             Every effect below this line is the fit moving, "
              "not a finding.")
    # DERIVED FROM `runs`, NOT RETYPED. This block and the `solo` line
    # below both used to hold their own hand-written list of group names,
    # and adding HISTORY on 16 Sep showed what that costs: "history only"
    # printed in the table at 60% of the full model while the sentence
    # underneath announced the best solo group was "per-change only" at
    # 35%, because the name was in one list and not the other. A summary
    # that can disagree with the table above it is worse than no summary.
    for name in (n for n in runs if n.startswith("no ")):
        lost = 1 - out[name]["pr_auc"] / base
        verdict = ("" if not controls else
                   "   <- BELOW THE NOISE FLOOR, read as nothing"
                   if abs(lost) <= floor else
                   # floor == 0 means every control was EXACTLY flat,
                   # which happens when the controls are features with no
                   # gain and few enough trees that dropping one changes
                   # nothing at all. Dividing by it printed "infx the
                   # floor". There is no multiple of zero; say that.
                   "   (no control moved at all — no floor to measure "
                   "against)" if floor == 0 else
                   f"   ({abs(lost) / floor:.1f}x the floor)")
        print(f"  removing {name.replace('no ', ''):<14} "
              f"costs {lost:>6.1%} of PR-AUC{verdict}")

    for name in controls:
        lost = 1 - out[name]["pr_auc"] / base
        print(f"  CONTROL  {name.replace('control: drop ', ''):<14} "
              f"moved {lost:>+6.1%}  (this feature had zero gain)")

    ps, rc = out["path shape only"]["pr_auc"], out["reachability only"]["pr_auc"]
    print(f"\n  path shape alone   {ps:.4f}  ({ps / base:.0%} of full) "
          f"— where the symbol is DEFINED")
    print(f"  reachability alone {rc:.4f}  ({rc / base:.0%} of full) "
          f"— how SHORT its public name is")
    if rc > ps:
        print("\n  Reachability beats definition-path shape. The model is")
        print("  closer to 'how easy is this to import' than to 'how deep is")
        print("  it buried' — which is the more defensible of the two, and")
        print("  also the one with a circularity worth stating: the usage")
        print("  index is full of short paths because that is what people")
        print("  write, so short-named symbols have more ways to match.")

    solo = max((n for n in runs if n.endswith(" only")),
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
