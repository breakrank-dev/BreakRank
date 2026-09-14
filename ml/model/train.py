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


# --------------------------------------------------- choosing a tree count

# Expanding-window boundaries inside TRAIN. Starts at 40% rather than 20%
# because a fold fitted on a fifth of the data stops early for reasons that
# have nothing to do with the final model.
CV_BOUNDS = [0.40, 0.55, 0.70, 0.85, 1.00]
MIN_TREES = 20


def cv_tree_count(full_train: pd.DataFrame, feats: list[str], label: str,
                  objective: str, cap: int = 600):
    """Pick the tree count by expanding-window CV inside train.

    THE PROBLEM. `split_valid` hands early stopping ONE slice — the newest
    20% of train — and that slice decides everything. Measured (§5.6): the
    chosen count ranged from 1 to 81 trees across seven cut dates, and the
    two worst results in the whole stability table were both 1-tree fits:
    label_scoped at 0.37x and the strict label at 0.57x, each LOSING to a
    baseline that sorts by download count. A stopping rule that can end
    training after one tree is not stopping early, it is not starting.

    It happens because the validation slice is not representative. Train
    runs 4.25% positive, that newest slice 6.39%, test 3.48% — validation
    is roughly twice as dense in positives as the half we actually score
    on, so "stopped improving" is measured against a distribution the
    model is never judged against.

    THE FIX, which is ordinary practice and not a trick: use several
    validation folds to CHOOSE a number, then refit on all of train with
    that number fixed. Each fold trains on everything before a date and
    validates on the window after it, so no fold ever sees its own future.
    The median across folds is what survives one bad slice — a single fold
    collapsing to 1 tree moves a median of four almost not at all, which
    is the entire point.

    TEST IS NEVER TOUCHED. Every fold lives inside train. The temporal
    train/test split is unchanged, and the tree count is a hyperparameter
    chosen on training data, exactly like any other.
    """
    d = pd.to_datetime(full_train["released_at"], errors="coerce")
    known = d.dropna()
    if known.empty:
        return cap, []

    iters = []
    for lo, hi in zip(CV_BOUNDS, CV_BOUNDS[1:]):
        c_lo, c_hi = known.quantile(lo), known.quantile(hi)
        sub = full_train[d <= c_lo].sort_values(GROUP)
        val = full_train[(d > c_lo) & (d <= c_hi)].sort_values(GROUP)
        # A fold with no positives on either side cannot rank anything and
        # would contribute a meaningless number to the median.
        if sub.empty or val.empty or not sub[label].sum() or not val[label].sum():
            continue
        try:
            m = fit_model(sub, val, feats, label, objective)
        except Exception:
            continue
        iters.append(int(getattr(m, "best_iteration_", None) or cap))

    if not iters:
        return cap, []
    n = int(np.median(iters))
    return max(n, MIN_TREES), iters


def fit_fixed(train: pd.DataFrame, feats: list[str], label: str,
              n_trees: int, objective: str = "lambdarank"):
    """Refit on ALL of train with the tree count already decided.

    No early stopping and no validation set, deliberately: the number was
    chosen by cv_tree_count() and re-deciding it here on a slice would put
    the §5.6 problem straight back.
    """
    common = dict(n_estimators=n_trees, learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.9, random_state=0, verbose=-1)
    if objective == "lambdarank":
        model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1],
                               **common)
        model.fit(train[feats], train[label], group=group_sizes(train))
    else:
        pos = max(int(train[label].sum()), 1)
        model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=(len(train) - pos) / pos, **common)
        model.fit(train[feats], train[label])
    return model


def fit_cv(full_train: pd.DataFrame, feats: list[str], label: str,
           objective: str = "lambdarank"):
    """cv_tree_count + fit_fixed. Returns (model, n_trees, fold_iters)."""
    n_trees, iters = cv_tree_count(full_train, feats, label, objective)
    model = fit_fixed(full_train.sort_values(GROUP), feats, label,
                      n_trees, objective)
    return model, n_trees, iters


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the BreakRank ranker.")
    ap.add_argument("--label", default="label",
                    choices=["label", "label_scoped", "label_alias"])
    ap.add_argument("--objective", default="lambdarank",
                    choices=["lambdarank", "binary"])
    ap.add_argument("--version", default=None,
                    help="model_run.version; defaults to label+objective")
    ap.add_argument("--stopping", default="cv", choices=["cv", "holdout"],
                    help="cv: pick the tree count by expanding-window CV "
                         "inside train, then refit on all of it. holdout: "
                         "the old single-slice early stopping, kept so the "
                         "two can be compared rather than asserted.")
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

    print(f"\nlabel {label}   objective {args.objective}   "
          f"stopping {args.stopping}")
    print(f"train {len(train):,} ({train[label].mean():.2%} pos)   "
          f"valid {len(valid):,} ({valid[label].mean():.2%} pos)   "
          f"test {len(test):,} ({test[label].mean():.2%} pos)")
    print(f"the holdout validation slice is train after {vcut.date()}")
    if valid[label].mean() > 1.5 * test[label].mean():
        print(f"** that slice is {valid[label].mean() / test[label].mean():.1f}x "
              "denser in positives than test — which is exactly\n"
              "** why --stopping cv exists (§5.6).")

    if args.stopping == "cv":
        model, best, folds = fit_cv(full_train, feats, label, args.objective)
        fit_on = full_train
        print(f"\nCV folds chose {folds} trees -> median {best}, "
              f"refitted on all {len(full_train):,} training rows")
        if len(set(folds)) > 1 and max(folds) > 5 * max(min(folds), 1):
            print(f"** the folds disagree by {max(folds) / max(min(folds), 1):.0f}x. "
                  "The median is doing real work here;\n** any single slice "
                  "could have handed back either end of that.")
    else:
        model = fit_model(train, valid, feats, label, args.objective)
        best = getattr(model, "best_iteration_", None) or 600
        fit_on = train
        print(f"\nstopped at {best} trees")
        if best < 50:
            print(f"** {best} trees is very few. Validation stopped improving "
                  "almost\n** immediately — either the signal is thin or the "
                  "validation half\n** is not representative. Check the "
                  "positive rates printed above.")
    print()

    scored = test.copy()
    scored["model"] = score_with(model, test, feats)
    scored = add_baseline_scores(fit_on, scored, label)

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
    # EVERY feature, not the top 8. The truncated version was actively
    # misleading on 14 Sep: `inherited_by` was absent from the printed
    # table, which reads as "the model ignores it" — and that flatly
    # contradicted an ablation saying its removal cost PR-AUC. The
    # contradiction was in the printing, not the model.
    #
    # A feature at ZERO gain is the interesting case, not the boring one:
    # the model was offered it and declined. That is worth seeing, and it
    # is exactly what head(8) hides once the feature set passes eight.
    imp["share"] = (imp["gain"] / max(imp["gain"].sum(), 1e-9) * 100).round(1)
    print(f"\nwhat the model actually used, all {len(feats)} features:")
    print(imp.to_string(index=False,
                        formatters={"gain": "{:,.1f}".format,
                                    "share": "{:>5.1f}%".format}))
    unused = imp[imp["gain"] <= 0]["feature"].tolist()
    if unused:
        print(f"\n** {len(unused)} feature(s) at ZERO gain: "
              f"{', '.join(unused)}")
        print("** The model was offered these and never split on them. If "
              "an\n** ablation says removing one changes PR-AUC, that "
              "difference is\n** the fit moving, not the feature working — "
              "treat it as the\n** table's noise floor and read every "
              "smaller gap as nothing.")
    if imp.iloc[0]["gain"] > 0.6 * imp["gain"].sum():
        print(f"\n** {imp.iloc[0]['feature']} is over 60% of total gain — "
              "the model is\n** close to a one-feature heuristic. Say so "
              "before anyone asks.")

    # PR-AUC's floor is the POSITIVE RATE, not 0.5 the way ROC-AUC's is.
    # Without it the number stored in model_run cannot be read by anyone
    # who was not in the room when it was produced: 0.35 is excellent at a
    # 3% positive rate and mediocre at 30%. The first run of this shipped
    # a model_run row with no floor in it, and reconstructing the number
    # afterwards meant reloading features.csv and re-deriving the split.
    # Record the floor beside the score.
    floor = float(test[label].mean())
    notes = (f"label={label} objective={args.objective} "
             f"stopping={args.stopping} trees={best} test_rows={len(test)} "
             f"positive_rate={floor:.4f} "
             f"best_baseline={best_base}:{bb:.4f}")

    # CARRY THE RANGE INTO THE ROW ITSELF. pr_auc here is ONE cut date, and
    # §5.6 measured that a single cut can sit anywhere in a band half as
    # wide as the number itself — this run's 0.3924 is near the top of
    # 0.309–0.438. The database is read by a website that will print
    # whatever it finds, and a context-free metric on a public page is the
    # exact failure this project spent two days documenting. `notes` is
    # free text and reaches the API unchanged, so the range rides along
    # with the number instead of living only in a notebook.
    stab = DATA / f"stability_{label}_{args.stopping}.csv"
    if stab.exists():
        st = pd.read_csv(stab)
        st = st[~st["skipped"].astype(bool)] if "skipped" in st else st
        if not st.empty:
            lift = st["lift_vs_pop"].astype(float)
            wins = int(st["beats_pop"].astype(bool).sum())
            notes += (f" | across {len(st)} cut dates: beats {best_base} "
                      f"{wins}/{len(st)}, lift median {lift.median():.2f}x "
                      f"min {lift.min():.2f}x max {lift.max():.2f}x")
        else:
            print("note: stability file has no usable splits; the model_run "
                  "row will carry a single-cut number with no range.")
    else:
        print(f"note: {stab} not found, so this model_run row will carry a "
              "single-cut\nnumber with no range. Run ml/model/stability.py "
              f"--label {label} --stopping {args.stopping} first.")

    run = {"version": version, **{k: round(v, 6) for k, v in m.items()},
           "notes": notes}
    (ART / "metrics.json").write_text(json.dumps(run, indent=2))
    print(f"\nartifacts/metrics.json — this is your model_run row:\n"
          f"{json.dumps(run, indent=2)}")

    print(f"\nHow to say this out loud, with all three numbers:\n"
          f"  PR-AUC {m['pr_auc']:.3f}, against a floor of {floor:.3f} "
          f"(the positive rate)\n"
          f"  and a best baseline of {bb:.3f} ({best_base}).\n"
          f"  That is {m['pr_auc'] / floor:.1f}x the floor and "
          f"{m['pr_auc'] / bb:.1f}x the strongest baseline.")
    if m["pr_auc"] <= bb:
        print("\n** The model does NOT beat its own baseline. Do not report "
              "this as a\n** result. Fix the model or report the baseline "
              "as the finding.")


if __name__ == "__main__":
    main()
