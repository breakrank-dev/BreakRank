"""
The bar the model has to clear. Run this BEFORE training anything.

    python ml/model/baselines.py
    python ml/model/baselines.py --label label

A learning-to-rank model that cannot beat "sort by download count" is not
a contribution, it is a slower way to sort by download count — and you
cannot know which you have unless you measure the dumb thing first. The
kill-date gate is stated in exactly these terms: the model must beat the
version-number baseline on PR-AUC.

Five baselines, weakest first:

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

  linear       a logistic regression on the ranker's own features, every
               one of them (F8, NOTES §29). Fitted on train only, like
               kind_prior. The ranker is a few hundred decision trees; this
               is one straight line through the same 17 numbers. If the
               line does as well, the trees are not what the result is
               made of, whatever the report says about U-shapes.

All five are scored on the TEST half only, with the same metrics the
ranker will use, so the comparison is like for like.
"""

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import assert_no_holdout  # noqa: E402
from ml.model.metrics import compare, evaluate, n_rankable  # noqa: E402

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"

GROUP = ["package", "version_from", "version_to"]

BUMP_SCORE = {"major": 3.0, "minor": 2.0, "patch": 1.0, "other": 0.0}

# ------------------------------------------------ the linear baseline (F8)

# The ranker's feature list, exactly (train.py builds the same one).
LINEAR_FEATS = NUMERIC + BOOLEAN + CATEGORICAL

# Counts and a rank with long tails: inherited_by is 0 on almost every row
# and 3,242 on a few (§10.2). On a straight line those few rows would set
# the scale for everything else, so these are log1p'd first. log1p is
# monotone, so it hands the line nothing it could not already express: a
# U-shape (F8's claim) stays out of its reach either way.
LOG_FEATS = ["inherited_by", "prior_breaks_in_module", "package_rank",
             "release_size", "package_churn"]


def _log1p_nonneg(x):
    return np.log1p(np.clip(x, 0, None))


def linear_model():
    """Logistic regression, L2 at scikit-learn's default strength, classes
    weighted to balance. Numbers are median-imputed and standardised, the
    long-tailed ones log1p'd first; kind and bump are one-hot, a value
    unseen in training scoring as all zeros. Nothing is tuned: it is a
    baseline, and the ranker's own settings are constants too (F7)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                       StandardScaler)

    logged = make_pipeline(SimpleImputer(strategy="median"),
                           FunctionTransformer(_log1p_nonneg),
                           StandardScaler())
    plain = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pre = ColumnTransformer([
        ("log", logged, LOG_FEATS),
        ("num", plain, [f for f in NUMERIC + BOOLEAN if f not in LOG_FEATS]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(pre, LogisticRegression(class_weight="balanced",
                                                 max_iter=5000))


def linear_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The 17 feature columns as the pipeline wants them: numbers and
    booleans as floats (a bool column, or one read back from CSV as text,
    both end up 0.0/1.0), kind and bump as strings."""
    x = df[LINEAR_FEATS].copy()
    for c in NUMERIC + BOOLEAN:
        col = x[c]
        if not pd.api.types.is_numeric_dtype(col):
            # Text, whatever string dtype pandas gave it: "True"/"False"
            # become 1/0, a number in text its value, anything else NaN.
            s = col.astype(str).str.strip().str.lower()
            col = s.map({"true": 1.0, "false": 0.0}).fillna(
                pd.to_numeric(s, errors="coerce"))
        x[c] = pd.to_numeric(col, errors="coerce").astype(float)
    for c in CATEGORICAL:
        x[c] = x[c].astype(str)
    return x


def linear_scores(train: pd.DataFrame, test: pd.DataFrame,
                  label: str) -> np.ndarray:
    """P(positive) from a line fitted on TRAIN ONLY, for every test row."""
    y = train[label].astype(int).to_numpy()
    if len(set(y)) < 2:
        raise ValueError(f"the linear baseline needs both classes in train; "
                         f"{label} has {int(y.sum())} positives in "
                         f"{len(y)} rows")
    with warnings.catch_warnings():
        # scikit-learn 1.6 hands scipy 1.18's L-BFGS an option it dropped
        # ("Unknown solver options: iprint"), once per fit. The fit is
        # unaffected; the warning is theirs, not a sign about the data.
        warnings.filterwarnings("ignore", message="Unknown solver options")
        model = linear_model().fit(linear_matrix(train), y)
    return model.predict_proba(linear_matrix(test))[:, 1]


def add_baseline_scores(train: pd.DataFrame, test: pd.DataFrame,
                        label: str) -> pd.DataFrame:
    test = test.copy()

    # Constant: no information at all. Ties are broken randomly in metrics.py,
    # which is what "no ranking" honestly means.
    test["griffe_all"] = 1.0

    # astype(str) before each map below. train.py and final_eval.py pass
    # `bump` and `kind` as pandas categories (prepare()), and on pandas 2.x
    # mapping a category can return a category, whose fillna() then raises
    # TypeError for any fill value that is not already one of its
    # categories. Measured on pandas 2.2.3: the old lines raised on every
    # prepared frame tried, so train.py could not run there at all. pandas
    # 3 does not raise. Scores are identical on both; this only stops the
    # crash.
    test["semver"] = test["bump"].astype(str).map(BUMP_SCORE).fillna(0.0)

    # rank 1 is the most downloaded, so invert it.
    test["popularity"] = -test["package_rank"].fillna(test["package_rank"].max())

    # Fitted on TRAIN ONLY. The global mean is the fallback for a kind the
    # training half never saw.
    rates = train.groupby("kind", observed=True)[label].mean()
    rates.index = rates.index.astype(str)
    test["kind_prior"] = (test["kind"].astype(str).map(rates)
                          .fillna(train[label].mean()))

    # F8. Also fitted on TRAIN ONLY, on the same rows the ranker was.
    test["linear"] = linear_scores(train, test, label)

    return test


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the dumb baselines.")
    ap.add_argument("--label", default="label_alias",
                    choices=["label", "label_scoped", "label_alias"],
                    help="which label to score against (label_alias "
                         "ships, NOTES §24)")
    args = ap.parse_args()

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = pd.read_csv(FEATURES)
    assert_no_holdout(df, "baselines.py")
    if args.label not in df.columns:
        sys.exit(f"no column {args.label} in {FEATURES}")

    train = df[df.split == "train"]
    test = df[df.split == "test"]
    if test[args.label].sum() == 0:
        sys.exit("the test half has no positives — nothing to measure")

    test = add_baseline_scores(train, test, args.label)

    names = ["griffe_all", "semver", "popularity", "kind_prior", "linear"]
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
    print(f"the line (logistic regression, the ranker's {len(LINEAR_FEATS)} "
          f"features): {results['linear']['pr_auc']:.4f}")
    print(f"\n** The ranker has to beat {results[best]['pr_auc']:.4f} PR-AUC "
          f"to be worth shipping. **")
    print(f"** The kill-date gate is the semver line: "
          f"{results['semver']['pr_auc']:.4f}. **")

    out = DATA / f"baselines_{args.label}.csv"
    pd.DataFrame(results).T.to_csv(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
