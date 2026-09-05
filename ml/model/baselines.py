"""
The bar the model has to clear. Run this BEFORE training anything.

    python ml/model/baselines.py
    python ml/model/baselines.py --label label_scoped

A learning-to-rank model that cannot beat "sort by download count" is not
a contribution, it is a slower way to sort by download count — and you
cannot know which you have unless you measure the dumb thing first. The
kill-date gate is stated in exactly these terms: the model must beat the
version-number baseline on PR-AUC.

Four baselines, weakest first:

  griffe_all   every change is equally important. This is the world
               WITHOUT BreakRank: 187 changes, no ordering, read them all.
               Its score is the floor, and it is roughly the positive rate.

  semver       trust the version number. major > minor > patch. This is
               what a careful developer already does, and beating it is
               the entire thesis: breaking changes hide in patch releases.

  popularity   rank by how downloaded the PACKAGE is, ignoring the change
               entirely. Sounds stupid; popularity priors are usually the
               hardest cheap baseline to beat, so it is here to keep us
               honest.

  kind_prior   score each change by how often its KIND was positive in
               TRAINING data. A one-feature model. Fitted on train only —
               fitting it on everything would be leakage wearing a
               baseline's clothes.

All four are scored on the TEST half only, with the same metrics the
ranker will use, so the comparison is like for like.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.model.metrics import compare, evaluate, n_rankable  # noqa: E402

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"

GROUP = ["package", "version_from", "version_to"]

BUMP_SCORE = {"major": 3.0, "minor": 2.0, "patch": 1.0, "other": 0.0}


def add_baseline_scores(train: pd.DataFrame, test: pd.DataFrame,
                        label: str) -> pd.DataFrame:
    test = test.copy()

    # Constant: no information at all. Ties are broken randomly in metrics.py,
    # which is what "no ranking" honestly means.
    test["griffe_all"] = 1.0

    test["semver"] = test["bump"].map(BUMP_SCORE).fillna(0.0)

    # rank 1 is the most downloaded, so invert it.
    test["popularity"] = -test["package_rank"].fillna(test["package_rank"].max())

    # Fitted on TRAIN ONLY. The global mean is the fallback for a kind the
    # training half never saw.
    rates = train.groupby("kind")[label].mean()
    test["kind_prior"] = test["kind"].map(rates).fillna(train[label].mean())

    return test


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the dumb baselines.")
    ap.add_argument("--label", default="label",
                    choices=["label", "label_scoped"],
                    help="which label to score against")
    args = ap.parse_args()

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = pd.read_csv(FEATURES)
    if args.label not in df.columns:
        sys.exit(f"no column {args.label} in {FEATURES}")

    train = df[df.split == "train"]
    test = df[df.split == "test"]
    if test[args.label].sum() == 0:
        sys.exit("the test half has no positives — nothing to measure")

    test = add_baseline_scores(train, test, args.label)

    names = ["griffe_all", "semver", "popularity", "kind_prior"]
    results = {n: evaluate(test, n, args.label) for n in names}

    print(f"\nlabel: {args.label}")
    print(f"train {len(train):,} rows ({train[args.label].mean():.2%} positive)"
          f"   test {len(test):,} rows ({test[args.label].mean():.2%} positive)")
    pairs = test.groupby(["package", "version_from", "version_to"])
    withpos = sum(1 for _, g in pairs if g[args.label].sum() > 0)
    rankable10 = n_rankable(test, args.label, 10)
    rankable20 = n_rankable(test, args.label, 20)
    print(f"{pairs.ngroups:,} version pairs in test   "
          f"{withpos:,} have a positive   "
          f"{rankable10:,} also have >10 changes")
    print("precision@10 and nDCG@20 are averaged over the rankable pairs only "
          f"({rankable10:,} and {rankable20:,}):\na release with 3 changes "
          "cannot be ranked wrong, so scoring it flatters everyone.\n")

    print(compare(results, baseline="semver"))

    flat = [n for n in names
            if test.groupby(GROUP)[n].nunique().max() <= 1]
    if flat:
        print(f"\nCANNOT RANK WITHIN A RELEASE: {', '.join(flat)}")
        print("Every change in one release shares that release's version bump\n"
              "and its package's download rank, so those scores are CONSTANT\n"
              "inside a version pair. They can tell you an upgrade is risky;\n"
              "they cannot tell you which of its 187 changes to read. They\n"
              "move PR-AUC (measured across releases) and are pinned to the\n"
              "no-ranking floor on precision@10 and nDCG.\n\n"
              "That is the gap the project exists to fill: inside one upgrade,\n"
              "only a per-CHANGE signal can order anything.")

    best = max(results, key=lambda n: results[n]["pr_auc"])
    floor = test[args.label].mean()
    print(f"\nrandom-guess PR-AUC would be {floor:.4f} (the positive rate).")
    print(f"strongest baseline: {best} at {results[best]['pr_auc']:.4f}")
    print(f"\n** The ranker has to beat {results[best]['pr_auc']:.4f} PR-AUC "
          f"to be worth shipping. **")
    print(f"** The kill-date gate is the semver line: "
          f"{results['semver']['pr_auc']:.4f}. **")

    out = DATA / f"baselines_{args.label}.csv"
    pd.DataFrame(results).T.to_csv(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
