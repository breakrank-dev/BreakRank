"""
The ranker. Everything before this was making the data honest.

    python ml/model/train.py
    python ml/model/train.py --label label_scoped
    python ml/model/train.py --objective binary

Reads  data/features.csv         (must have a `split` column)
Writes artifacts/ranker.txt      the model
       artifacts/metrics.json    a `model_run` row, ready for the database
       artifacts/importance.csv  which features did the work

LEARNING TO RANK, NOT CLASSIFICATION, and the difference is the product.
A classifier answers "will this change break someone?" one row at a time.
The user is looking at 187 changes from one upgrade and wants the five
worth reading FIRST. That is an ordering problem inside a group, so the
groups are version pairs and the objective is lambdarank.

`--objective binary` trains a plain classifier instead. Worth running: if
the classifier wins, say so and ship it. Assuming the fancier tool is
better is how people end up defending a worse model in a viva.

VALIDATION IS TEMPORAL TOO. Early stopping needs data the model has not
seen, and taking it randomly out of train would leak the future backwards
through the stopping rule — a subtle version of the same mistake the
train/test split exists to avoid. So the newest slice of TRAIN becomes
validation, and test is never touched until the end.
"""

import argparse
import json
import pathlib
import sys
import warnings

import lightgbm as lgb

# Deprecation chatter from the sklearn wrapper, once per fit, drowning the
# output we actually read. The API we use still works on 4.x.
warnings.filterwarnings("ignore", module="lightgbm")
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import compare, evaluate, n_rankable  # noqa: E402

DATA = pathlib.Path("data")
ART = pathlib.Path("artifacts")
FEATURES = DATA / "features.csv"
GROUP = ["package", "version_from", "version_to"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """lambdarank needs group sizes, and rows must already be contiguous."""
    return df.groupby(GROUP, sort=False).size().to_numpy()


def split_valid(train: pd.DataFrame, frac: float = 0.2):
    """Newest slice of train becomes validation. Temporal, like everything."""
    d = pd.to_datetime(train["released_at"], errors="coerce")
    cutoff = d.dropna().quantile(1 - frac)
    is_valid = d > cutoff
    return train[~is_valid], train[is_valid], cutoff


def fit_model(train: pd.DataFrame, valid: pd.DataFrame, feats: list[str],
              label: str, objective: str = "lambdarank"):
    """Fit one model. Split out so ablate.py can refit with fewer features."""
    common = dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.9, random_state=0, verbose=-1)

    if objective == "lambdarank":
        model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1],
                               **common)
        fit_kw = dict(group=group_sizes(train),
                      eval_group=[group_sizes(valid)], eval_at=[10])
    else:
        # scale_pos_weight matters at 3% positives: without it the model can
        # minimise loss by predicting "nobody cares" for every row.
        pos = max(int(train[label].sum()), 1)
        model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=(len(train) - pos) / pos, **common)
        fit_kw = {}

    model.fit(train[feats], train[label],
              eval_set=[(valid[feats], valid[label])],
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(0)],
              **fit_kw)
    return model


def score_with(model, frame: pd.DataFrame, feats: list[str]):
    raw = model.predict(frame[feats])
    return raw[:, 1] if getattr(raw, "ndim", 1) > 1 else raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the BreakRank ranker.")
    ap.add_argument("--label", default="label",
                    choices=["label", "label_scoped"])
    ap.add_argument("--objective", default="lambdarank",
                    choices=["lambdarank", "binary"])
    ap.add_argument("--version", default=None,
                    help="model_run.version; defaults to label+objective")
    args = ap.parse_args()
    label = args.label
    version = args.version or f"{args.objective}-{label}"

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    full_train = df[df.split == "train"].sort_values(GROUP)
    test = df[df.split == "test"].sort_values(GROUP)
    train, valid, vcut = split_valid(full_train)
    train, valid = train.sort_values(GROUP), valid.sort_values(GROUP)

    print(f"\nlabel {label}   objective {args.objective}")
    print(f"train {len(train):,} ({train[label].mean():.2%} pos)   "
          f"valid {len(valid):,} ({valid[label].mean():.2%} pos)   "
          f"test {len(test):,} ({test[label].mean():.2%} pos)")
    print(f"validation is everything in train after {vcut.date()}\n")

    model = fit_model(train, valid, feats, label, args.objective)
    best = getattr(model, "best_iteration_", None) or 600
    print(f"stopped at {best} trees")
    if best < 50:
        print(f"** {best} trees is very few. Validation stopped improving "
              "almost\n** immediately — either the signal is thin or the "
              "validation half\n** is not representative. Check the positive "
              "rates printed above.")
    print()

    scored = test.copy()
    scored["model"] = score_with(model, test, feats)
    scored = add_baseline_scores(train, scored, label)

    names = ["model", "popularity", "kind_prior", "semver", "griffe_all"]
    results = {n: evaluate(scored, n, label) for n in names}

    r10, r20 = n_rankable(test, label, 10), n_rankable(test, label, 20)
    print(f"test: {r10} pairs rankable at 10, {r20} at 20\n")
    print(compare(results, baseline="semver"))

    m, best_base = results["model"], max(
        (n for n in names if n != "model"), key=lambda n: results[n]["pr_auc"])
    bb = results[best_base]["pr_auc"]
    print(f"\nmodel PR-AUC {m['pr_auc']:.4f}   "
          f"best baseline ({best_base}) {bb:.4f}   "
          f"lift {m['pr_auc'] / bb if bb else float('nan'):.2f}x")
    print(f"kill-date gate (semver {results['semver']['pr_auc']:.4f}): "
          f"{'PASSED' if m['pr_auc'] > results['semver']['pr_auc'] else 'NOT PASSED'}")
    if m["pr_auc"] <= bb:
        print(f"\n** {best_base} still wins. Do not ship this. A ranker that "
              f"loses to\n** a one-line heuristic is a finding, not a failure "
              "— report it and\n** fix the features before touching the "
              "hyperparameters.")

    ART.mkdir(exist_ok=True)
    model.booster_.save_model(str(ART / "ranker.txt")) if hasattr(
        model, "booster_") else None

    imp = (pd.DataFrame({"feature": feats,
                         "gain": model.booster_.feature_importance("gain")})
             .sort_values("gain", ascending=False))
    imp.to_csv(ART / "importance.csv", index=False)
    print("\nwhat the model actually used:")
    print(imp.head(8).to_string(index=False))
    if imp.iloc[0]["gain"] > 0.6 * imp["gain"].sum():
        print(f"\n** {imp.iloc[0]['feature']} is over 60% of total gain — "
              "the model is\n** close to a one-feature heuristic. Say so "
              "before anyone asks.")

    run = {"version": version, **{k: round(v, 6) for k, v in m.items()},
           "notes": f"label={label} objective={args.objective} "
                    f"trees={best} test_rows={len(test)} "
                    f"best_baseline={best_base}:{bb:.4f}"}
    (ART / "metrics.json").write_text(json.dumps(run, indent=2))
    print(f"\nartifacts/metrics.json — this is your model_run row:\n"
          f"{json.dumps(run, indent=2)}")


if __name__ == "__main__":
    main()
