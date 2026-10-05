"""
The ranker. Everything before this was making the data honest.

    python ml/model/train.py
    python ml/model/train.py --label label
    python ml/model/train.py --objective binary
    python ml/model/train.py --relevance graded
    python ml/model/train.py --tuning fixed

The default label is label_alias, chosen 30 Sep by the rule in NOTES §24.
The default relevance is SHIPPED_RELEVANCE below; NOTES §25 says how
item 3 decides it.

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

WHAT THE RANKER IS TAUGHT (--relevance, F2). `binary` tells lambdarank
that every positive is worth the same: a change one package uses counts
as much as one forty packages use. `graded` gives each positive a grade
from 1 to 4 by how many packages use it (grades() below), so the ranker
is pushed to put the widest breaks first. Only the training target
changes. Which rows are positive, the test halves, and PR-AUC,
precision@10 and nDCG@20 all still come from the 0/1 label, so the two
are scored on the same exam.
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
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import HOLDOUT_START, assert_no_holdout  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import (DRAWS, compare, evaluate,  # noqa: E402
                              intervals, n_rankable, ndcg_at_k)

DATA = pathlib.Path("data")
ART = pathlib.Path("artifacts")
FEATURES = DATA / "features.csv"
GROUP = ["package", "version_from", "version_to"]


# ------------------------------------------ what the ranker is taught (F2)

RELEVANCE = ["binary", "graded"]

# What train.py, stability.py, final_eval.py and ablate.py use when
# --relevance is not given: the shipped model's. One constant, so the four
# cannot drift apart. Item 3 decides whether it changes (NOTES §25).
#
# The fit functions below default to "binary" instead, on purpose.
# fold_effect.py, label_blindspot.py and item2_effects.py re-measure
# results that were about the binary model, and have to keep doing so
# whatever ships.
SHIPPED_RELEVANCE = "binary"

# The count each label is built from (ml/features/labels.py): label is
# user_count > 0, label_alias is alias_user_count > 0, label_scoped is
# scoped_user_count > 0. A grade is read from the same count.
COUNT_OF = {"label": "user_count", "label_scoped": "scoped_user_count",
            "label_alias": "alias_user_count"}

# The gain of grades 0 to 4 is 2**grade - 1: LightGBM's own default, and
# the usual nDCG gain. A change 20 or more packages use is worth 15 of one
# that a single package uses.
GRADE_GAIN = [0, 1, 3, 7, 15]


def grades(df: pd.DataFrame, label: str) -> pd.Series:
    """0 for a negative. A positive gets 1 to 4 by how many packages use
    it, on a log scale, ceil(ln(1 + n)):

        1 package -> 1    2 to 6 -> 2    7 to 19 -> 3    20 or more -> 4

    A log scale because, for ranking, one user against five is a bigger
    difference than 100 against 104. The label decides WHICH rows count
    and the count only decides how much: a positive is never graded 0,
    whatever its count says, and a negative is never graded above it."""
    n = pd.to_numeric(df[COUNT_OF[label]], errors="coerce").fillna(0)
    g = np.ceil(np.log1p(n.clip(lower=0))).clip(1, 4).astype(int)
    return g.where(df[label].astype(int) == 1, 0).astype(int)


def graded_gain(df: pd.DataFrame, label: str) -> np.ndarray:
    """Each row's gain under graded relevance: 0, 1, 3, 7 or 15."""
    return np.asarray(GRADE_GAIN)[grades(df, label).to_numpy()]


def target(df: pd.DataFrame, label: str, relevance: str) -> pd.Series:
    """What LightGBM is handed as y: the 0/1 label, or its grades."""
    return grades(df, label) if relevance == "graded" else df[label]


# ------------------------------------------- how the trees are sized (F7)

TUNING = ["fixed", "cv"]

# What train.py, stability.py, final_eval.py and ablate.py use when
# --tuning is not given: the shipped model's.
#
#   fixed  one setting (FIXED_PARAMS). Its tree count is the median of
#          CV folds that stop on LightGBM's nDCG@10, clamped at 20 trees.
#          What shipped from 5 Sep to 2 Oct.
#   cv     the setting AND the tree count chosen on the same CV folds,
#          each fold stopping on PR-AUC; no clamp (cv_tune below).
#
# "cv" since 2 Oct, by the rule fixed before the run (NOTES §30.1): at the
# seven dates it beat popularity at all 7, worst lift 2.20x against
# fixed's 2.30x, median 5.20x against 4.08x (§30.2).
SHIPPED_TUNING = "cv"

# The setting every model has used since 5 Sep.
FIXED_PARAMS = dict(num_leaves=31, learning_rate=0.05, min_child_samples=30)

# What --tuning cv chooses among: leaves per tree, the smallest leaf, the
# learning rate. 18 settings, simplest first (fewest leaves, largest
# leaves), so on a tie the simpler one wins. FIXED_PARAMS is one of them,
# so tuning can keep the shipped setting and change only the tree count.
GRID = [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc)
        for nl in (7, 15, 31) for mc in (100, 30, 10) for lr in (0.05, 0.02)]
TUNE_CAP = 1000      # most trees a tuned fold may grow
STOP_ROUNDS = 60     # the early-stopping patience, as in fit_model


def tuning_problem(tuning: str, relevance: str = "binary",
                   stopping: str = "cv") -> str | None:
    """Why this run cannot use this tuning, or None if it can."""
    if tuning not in TUNING:
        return f"tuning must be one of {TUNING}, not {tuning!r}"
    if tuning == "cv" and relevance == "graded":
        return ("--tuning cv stops each fold on PR-AUC, a 0/1 measure, and "
                "is binary relevance only.\nUse --relevance binary, or "
                "--tuning fixed.")
    if tuning == "cv" and stopping != "cv":
        return ("--tuning cv chooses the setting on the CV folds, so it "
                "needs --stopping cv.")
    return None


def relevance_problem(df: pd.DataFrame, label: str, objective: str,
                      relevance: str) -> str | None:
    """Why this run cannot train with this relevance, or None if it can."""
    if relevance not in RELEVANCE:
        return f"relevance must be one of {RELEVANCE}, not {relevance!r}"
    if relevance == "graded" and objective != "lambdarank":
        return ("graded relevance is a ranking target, and --objective "
                "binary trains a classifier.\nUse --objective lambdarank.")
    if relevance == "graded" and COUNT_OF[label] not in df.columns:
        return (f"graded relevance reads {COUNT_OF[label]}, and this "
                "features.csv has no such column.\nRe-run "
                "ml/features/labels.py, then ml/features/build.py.")
    return None


def _check(df: pd.DataFrame, label: str, objective: str,
           relevance: str) -> None:
    """relevance_problem, raised. Called before any fit, and before
    cv_tree_count's try block, so a graded run can never fail quietly
    into a default tree count."""
    problem = relevance_problem(df, label, objective, relevance)
    if problem:
        raise ValueError(problem)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def stability_name(label: str, stopping: str, objective: str = "lambdarank",
                   relevance: str = "binary", at: bool = False,
                   tuning: str = "fixed") -> str:
    """The file a sweep writes and train.py quotes, one name for both.

    The shipped kind of sweep keeps the name it always had, so every file
    written so far is still the right one. A sweep of another objective
    gets its own: until 2 Oct `--objective binary` wrote over the
    lambdarank file, so F8's classifier sweep would have erased §27.2's.
    A tuned sweep (F7) gets _tuned, after _graded and before _at."""
    return (f"stability_{label}_{stopping}"
            + ("" if objective == "lambdarank" else f"_{objective}")
            + ("_graded" if relevance == "graded" else "")
            + ("_tuned" if tuning == "cv" else "")
            + ("_at" if at else "") + ".csv")


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
              label: str, objective: str = "lambdarank",
              relevance: str = "binary"):
    """Fit one model. Split out so ablate.py can refit with fewer features.

    With relevance="graded" the validation rows are graded too, so early
    stopping watches the same target the trees are fitted to."""
    _check(train, label, objective, relevance)
    common = dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.9, random_state=0, verbose=-1)

    if objective == "lambdarank":
        model = lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=GRADE_GAIN if relevance == "graded" else [0, 1],
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

    model.fit(train[feats], target(train, label, relevance),
              eval_set=[(valid[feats], target(valid, label, relevance))],
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(0)],
              **fit_kw)
    return model


def score_with(model, frame: pd.DataFrame, feats: list[str]):
    """One score per row, from whichever model this is.

    A classifier's predict() returns CLASS LABELS, 0 or 1, not
    probabilities, and a column of 0s and 1s ranks nothing: every row in a
    class ties. Until 2 Oct this function scored `--objective binary` with
    those labels, so the classifier-versus-ranker comparison F25 asks for
    was never possible, and the binary objective looked far worse than it
    is. On the test fixture (scripts/test_holdout.py's, cut at 15 May) a
    20-tree classifier scores PR-AUC 0.150 by its probabilities and 0.097
    by its labels, against a floor of 0.084. Its probabilities are its
    scores.
    The ranker has no predict_proba and its predict() is already a score,
    so nothing the shipped model reports changes."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame[feats])[:, 1]
    raw = model.predict(frame[feats])
    return raw[:, 1] if getattr(raw, "ndim", 1) > 1 else raw


# --------------------------------------------------- choosing a tree count

# Expanding-window boundaries inside TRAIN. Starts at 40% rather than 20%
# because a fold fitted on a fifth of the data stops early for reasons that
# have nothing to do with the final model.
CV_BOUNDS = [0.40, 0.55, 0.70, 0.85, 1.00]
MIN_TREES = 20


def cv_tree_count(full_train: pd.DataFrame, feats: list[str], label: str,
                  objective: str, cap: int = 600, relevance: str = "binary"):
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

    Under graded relevance each fold stops on graded nDCG@10, the target
    it is fitting. A fold still needs a 0/1 positive on both sides.
    """
    # Before the try below, which skips a fold that fails to fit: a graded
    # run with no count column must stop here, not fail every fold quietly
    # and come back with the 600-tree cap.
    _check(full_train, label, objective, relevance)

    iters = []
    for sub, val in cv_folds(full_train, label):
        try:
            m = fit_model(sub, val, feats, label, objective, relevance)
        except Exception:
            continue
        iters.append(int(getattr(m, "best_iteration_", None) or cap))

    if not iters:
        return cap, []
    n = int(np.median(iters))
    return max(n, MIN_TREES), iters


def cv_folds(full_train: pd.DataFrame, label: str):
    """The expanding folds inside TRAIN, as (fit on, validate on) frames,
    each sorted by upgrade: everything up to a date, then the window after
    it. cv_tree_count and cv_tune use the same ones."""
    d = pd.to_datetime(full_train["released_at"], errors="coerce")
    known = d.dropna()
    if known.empty:
        return
    for lo, hi in zip(CV_BOUNDS, CV_BOUNDS[1:]):
        c_lo, c_hi = known.quantile(lo), known.quantile(hi)
        sub = full_train[d <= c_lo].sort_values(GROUP)
        val = full_train[(d > c_lo) & (d <= c_hi)].sort_values(GROUP)
        # A fold with no positives on either side cannot rank anything and
        # would contribute a meaningless number to the median.
        if sub.empty or val.empty or not sub[label].sum() or not val[label].sum():
            continue
        yield sub, val


def fit_stopping_on_ap(sub: pd.DataFrame, val: pd.DataFrame,
                       feats: list[str], label: str, objective: str,
                       params: dict):
    """One tuned fold: grow up to TUNE_CAP trees with `params`, and stop
    when PR-AUC on the validation window has not improved for STOP_ROUNDS.

    WHY PR-AUC AND NOT nDCG@10 (F7). LightGBM scores an upgrade with no
    positive at nDCG 1.0, whatever the model does to it (measured 2 Oct:
    an all-negative group's implied score is exactly 1.0), and most
    upgrades have no positive. So the number fit_model stops on is mostly
    a constant, and "stopped improving" is decided by a handful of
    upgrades: the folds' tree counts for the shipped model ran from 2 to
    46 on one training set (§24.3). PR-AUC is pooled over the window's
    rows, so an all-negative upgrade still counts, and it is the number
    every result here is judged on. LightGBM's own average_precision is
    used for speed; it equals scikit-learn's (checked 2 Oct)."""
    common = dict(n_estimators=TUNE_CAP, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.9, random_state=0, verbose=-1,
                  metric="average_precision", **params)
    y, yv = sub[label].astype(int), val[label].astype(int)
    if objective == "lambdarank":
        model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1],
                               **common)
        fit_kw = dict(group=group_sizes(sub), eval_group=[group_sizes(val)])
    else:
        pos = max(int(y.sum()), 1)
        model = lgb.LGBMClassifier(objective="binary",
                                   scale_pos_weight=(len(sub) - pos) / pos,
                                   **common)
        fit_kw = {}
    model.fit(sub[feats], y, eval_set=[(val[feats], yv)],
              callbacks=[lgb.early_stopping(STOP_ROUNDS, verbose=False),
                         lgb.log_evaluation(0)],
              **fit_kw)
    return model


def choose_setting(rows: list[dict]) -> dict:
    """The winning row of cv_tune's table: the best median score, and on a
    tie the one earliest in GRID, which is the simpler one."""
    return max(rows, key=lambda r: (r["score"], -r["order"]))


def cv_tune(full_train: pd.DataFrame, feats: list[str], label: str,
            objective: str, grid: list[dict] | None = None):
    """--tuning cv: choose the setting and the tree count on the folds.

    For every setting in GRID, every fold grows trees on its past and
    stops on PR-AUC in its window (fit_stopping_on_ap). The fold's score
    is that PR-AUC over the window's floor, its positive rate, so folds
    with different floors count alike; the setting's score is the median
    over folds. The best setting wins, and its tree count is the median of
    the trees its folds stopped at. NO CLAMP: if the data says 8 trees,
    that is the answer (F7). TEST IS NEVER TOUCHED; every fold lives
    inside train, as in cv_tree_count.

    Returns (setting, n_trees, that setting's fold tree counts, the table
    of every setting tried)."""
    grid = GRID if grid is None else grid
    folds = list(cv_folds(full_train, label))
    rows = []
    for order, params in enumerate(grid):
        trees, scores = [], []
        for sub, val in folds:
            m = fit_stopping_on_ap(sub, val, feats, label, objective, params)
            best = int(getattr(m, "best_iteration_", None) or TUNE_CAP)
            yv = val[label].astype(int).to_numpy()
            ap = float(average_precision_score(yv, score_with(m, val, feats)))
            trees.append(best)
            scores.append(ap / yv.mean())
        if scores:
            rows.append({"order": order, **params,
                         "score": float(np.median(scores)),
                         "trees": max(int(np.median(trees)), 1),
                         "fold_trees": trees})
    if not rows:
        # No usable fold: what cv_tree_count does with none, the shipped
        # setting at its cap.
        return dict(FIXED_PARAMS), 600, [], []
    best = choose_setting(rows)
    setting = {k: best[k] for k in FIXED_PARAMS}
    return setting, best["trees"], best["fold_trees"], rows


def setting_text(params: dict) -> str:
    """A setting as one short token: leaves/learning rate/smallest leaf."""
    return (f"{params['num_leaves']}/{params['learning_rate']:g}/"
            f"{params['min_child_samples']}")


def fit_fixed(train: pd.DataFrame, feats: list[str], label: str,
              n_trees: int, objective: str = "lambdarank",
              relevance: str = "binary", params: dict | None = None):
    """Refit on ALL of train with the tree count already decided.

    No early stopping and no validation set, deliberately: the number was
    chosen by cv_tree_count() and re-deciding it here on a slice would put
    the §5.6 problem straight back. `params` replaces FIXED_PARAMS with
    the setting cv_tune chose; left out, the model is the shipped one.
    """
    _check(train, label, objective, relevance)
    common = dict(n_estimators=n_trees, **FIXED_PARAMS, subsample=0.9,
                  subsample_freq=1, colsample_bytree=0.9, random_state=0,
                  verbose=-1)
    if params:
        common.update(params)
    if objective == "lambdarank":
        model = lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=GRADE_GAIN if relevance == "graded" else [0, 1],
            **common)
        model.fit(train[feats], target(train, label, relevance),
                  group=group_sizes(train))
    else:
        pos = max(int(train[label].sum()), 1)
        model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=(len(train) - pos) / pos, **common)
        model.fit(train[feats], train[label])
    return model


def fit_cv(full_train: pd.DataFrame, feats: list[str], label: str,
           objective: str = "lambdarank", relevance: str = "binary",
           tuning: str = "fixed"):
    """Choose the tree count (and with tuning="cv" the setting) on the
    folds, then refit on all of train. Returns (model, n_trees,
    fold_iters). A tuned model also carries `tuned_`: the setting chosen
    and the table of every setting tried.

    tuning defaults to "fixed" here, like relevance, so the scripts that
    re-measure the shipped model keep doing so whatever SHIPPED_TUNING
    says; train.py, stability.py and final_eval.py pass it explicitly."""
    problem = tuning_problem(tuning, relevance)
    if problem:
        raise ValueError(problem)
    if tuning == "cv":
        _check(full_train, label, objective, relevance)
        params, n_trees, iters, table = cv_tune(full_train, feats, label,
                                                objective)
        model = fit_fixed(full_train.sort_values(GROUP), feats, label,
                          n_trees, objective, relevance, params=params)
        model.tuned_ = {"params": params, "table": table}
        return model, n_trees, iters
    n_trees, iters = cv_tree_count(full_train, feats, label, objective,
                                   relevance=relevance)
    model = fit_fixed(full_train.sort_values(GROUP), feats, label,
                      n_trees, objective, relevance)
    return model, n_trees, iters


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the BreakRank ranker.")
    ap.add_argument("--label", default="label_alias",
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
    ap.add_argument("--relevance", default=SHIPPED_RELEVANCE,
                    choices=RELEVANCE,
                    help="binary: every positive counts the same. graded: "
                         "a positive counts more the more packages use it "
                         "(F2, NOTES §25).")
    ap.add_argument("--tuning", default=SHIPPED_TUNING, choices=TUNING,
                    help="fixed: one setting, its tree count from folds "
                         "stopping on nDCG@10, clamped at 20. cv: the "
                         "setting and tree count chosen on the folds, each "
                         "stopping on PR-AUC, no clamp (F7, NOTES §30).")
    args = ap.parse_args()
    label = args.label
    graded = args.relevance == "graded"
    tuned = args.tuning == "cv"
    # A graded or tuned model is a different model, so it gets its own
    # model_run row. The shipped name is unchanged, so earlier runs keep
    # theirs.
    version = args.version or (f"{args.objective}-{label}"
                               + ("-graded" if graded else "")
                               + ("-tuned" if tuned else ""))

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    # Every number this script prints is a DEV number. The frozen holdout
    # (ml/holdout.py) is not in features.csv, and a stale file that still
    # holds it stops here rather than being scored.
    assert_no_holdout(df, "train.py")
    problem = (relevance_problem(df, label, args.objective, args.relevance)
               or tuning_problem(args.tuning, args.relevance, args.stopping))
    if problem:
        sys.exit(f"train.py: {problem}")
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    full_train = df[df.split == "train"].sort_values(GROUP)
    test = df[df.split == "test"].sort_values(GROUP)
    train, valid, vcut = split_valid(full_train)
    train, valid = train.sort_values(GROUP), valid.sort_values(GROUP)

    print(f"\nlabel {label}   objective {args.objective}   "
          f"relevance {args.relevance}   stopping {args.stopping}   "
          f"tuning {args.tuning}")
    print(f"train {len(train):,} ({train[label].mean():.2%} pos)   "
          f"valid {len(valid):,} ({valid[label].mean():.2%} pos)   "
          f"test {len(test):,} ({test[label].mean():.2%} pos)")
    # F12. Rows are not independent observations: a symbol's parameter
    # rows share its label, and a release's changes rise and fall together.
    n_changes = len(test.drop_duplicates(GROUP + ["symbol"]))
    n_upgrades = len(test.drop_duplicates(GROUP))
    print(f"test's {len(test):,} rows are {n_changes:,} changes (a symbol in "
          f"one upgrade) in {n_upgrades:,} upgrades")
    print(f"the holdout validation slice is train after {vcut.date()}")
    if valid[label].mean() > 1.5 * test[label].mean():
        print(f"** that slice is {valid[label].mean() / test[label].mean():.1f}x "
              "denser in positives than test — which is exactly\n"
              "** why --stopping cv exists (§5.6).")
    if graded:
        g = grades(full_train, label)
        share = g[g > 0].value_counts(normalize=True).reindex(
            [1, 2, 3, 4], fill_value=0)
        print("training positives by grade:   "
              + "   ".join(f"{k} ({w}) {v:.0%}" for (k, v), w in
                           zip(share.items(), ["1 package", "2-6", "7-19",
                                               "20+"])))

    if args.stopping == "cv":
        model, best, folds = fit_cv(full_train, feats, label, args.objective,
                                    relevance=args.relevance,
                                    tuning=args.tuning)
        fit_on = full_train
        if tuned:
            table = model.tuned_["table"]
            print(f"\ntuned on the same folds: {len(table)} settings "
                  "(leaves/learning rate/smallest leaf), each fold stopping "
                  "on PR-AUC.\nbest by median PR-AUC over each fold's "
                  "floor:")
            for r in sorted(table, key=lambda r: -r["score"])[:5]:
                print(f"  {setting_text(r):<12}{r['score']:>7.3f}x floor"
                      f"   folds stopped at {r['fold_trees']}")
            print(f"chosen {setting_text(model.tuned_['params'])}: its folds "
                  f"stopped at {folds} -> median {best}, no clamp, "
                  f"refitted on all {len(full_train):,} training rows")
        else:
            print(f"\nCV folds chose {folds} trees -> median {best}, "
                  f"refitted on all {len(full_train):,} training rows")
        if len(set(folds)) > 1 and max(folds) > 5 * max(min(folds), 1):
            print(f"** the folds disagree by {max(folds) / max(min(folds), 1):.0f}x. "
                  "The median is doing real work here;\n** any single slice "
                  "could have handed back either end of that.")
    else:
        model = fit_model(train, valid, feats, label, args.objective,
                          relevance=args.relevance)
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

    names = ["model", "linear", "popularity", "kind_prior", "semver",
             "griffe_all"]
    results = {n: evaluate(scored, n, label) for n in names}

    r10, r20 = n_rankable(test, label, 10), n_rankable(test, label, 20)
    print(f"test: {r10} pairs rankable at 10, {r20} at 20\n")
    print(compare(results, baseline="semver"))

    # The measure graded relevance aims at, printed for either kind of
    # model so the two can be read side by side: nDCG@20 over the same
    # upgrades, a change's gain 1, 3, 7 or 15 by its grade instead of 1.
    if COUNT_OF[label] in scored:
        weighed = scored.assign(_gain=graded_gain(scored, label))
        gn = {n: ndcg_at_k(weighed, n, label, 20, gain="_gain")
              for n in ("model", "kind_prior", "popularity")}
        print(f"\nnDCG@20 with graded gains (F2), the same {r20} upgrades:  "
              + "   ".join(f"{n} {v:.4f}" for n, v in gn.items()))

    m, best_base = results["model"], max(
        (n for n in names if n != "model"), key=lambda n: results[n]["pr_auc"])
    bb = results[best_base]["pr_auc"]
    print(f"\nmodel PR-AUC {m['pr_auc']:.4f}   "
          f"best baseline ({best_base}) {bb:.4f}   "
          f"lift {m['pr_auc'] / bb if bb else float('nan'):.2f}x")
    print(f"kill-date gate (semver {results['semver']['pr_auc']:.4f}): "
          f"{'PASSED' if m['pr_auc'] > results['semver']['pr_auc'] else 'NOT PASSED'}")
    if m["pr_auc"] <= bb and best_base == "linear":
        print("\n** The line wins this split: a logistic regression on the "
              "same features\n** beats the trees. One split is not the "
              "answer; scripts/item7_linear.py\n** measures it at seven "
              "dates and NOTES §29.1 says what each outcome means.")
    elif m["pr_auc"] <= bb:
        print(f"\n** {best_base} still wins. Do not ship this. A ranker that "
              f"loses to\n** a one-line heuristic is a finding, not a failure "
              "— report it and\n** fix the features before touching the "
              "hyperparameters.")

    # F11. How far each number could move on another draw of test
    # upgrades, the models held fixed (metrics.intervals says why whole
    # upgrades, and why no refit). The range across cut dates is
    # stability.py's; this is the other kind of uncertainty.
    ci = intervals(scored, label, "model", "popularity")
    pop = results["popularity"]["pr_auc"]
    lift_pop = m["pr_auc"] / pop if pop else float("nan")
    print(f"\n95% intervals, from {DRAWS:,} resamples of the test half's "
          f"{ci['upgrades']:,} upgrades (the models are not refitted):")
    for name, v, (lo, hi), fmt, n in (
            ("PR-AUC", m["pr_auc"], ci["pr_auc"], "{:.3f}", None),
            ("popularity PR-AUC", pop, ci["baseline_pr_auc"], "{:.3f}", None),
            ("lift over popularity", lift_pop, ci["lift"], "{:.2f}x", None),
            ("precision@10", m["precision_at_10"], ci["precision_at_10"],
             "{:.3f}", r10),
            ("nDCG@20", m["ndcg_at_20"], ci["ndcg_at_20"], "{:.3f}", r20)):
        print(f"  {name:<22}{fmt.format(v):>7}   {fmt.format(lo)} to "
              f"{fmt.format(hi)}" + (f"   over {n} upgrades" if n else ""))
    if not ci["lift"][0] > 1:
        print("** The lift's interval reaches 1.0x: on some draws of test "
              "upgrades this\n** model does not beat popularity. Say so "
              "beside the number.")

    # F8. The same again against the line, and against whichever baseline
    # won this split if that was neither. Each draw scores both models on
    # the same upgrades, so these are paired: the interval of the ratio,
    # not two intervals held side by side.
    lin = results["linear"]["pr_auc"]
    lift_lin = m["pr_auc"] / lin if lin else float("nan")
    ci_lin = intervals(scored, label, "model", "linear")
    print(f"  {'lift over the line':<22}{lift_lin:>6.2f}x   "
          f"{ci_lin['lift'][0]:.2f}x to {ci_lin['lift'][1]:.2f}x   "
          f"(logistic regression, the same {len(feats)} features)")
    if best_base not in ("popularity", "linear"):
        ci_best = intervals(scored, label, "model", best_base)
        print(f"  {'lift over ' + best_base:<22}"
              f"{m['pr_auc'] / bb:>6.2f}x   {ci_best['lift'][0]:.2f}x to "
              f"{ci_best['lift'][1]:.2f}x   (the best baseline here)")
    if not ci_lin["lift"][0] > 1:
        print("** The line's interval reaches 1.0x: on some draws of test "
              "upgrades the\n** trees do not beat a logistic regression on "
              "their own features.\n** NOTES §29 says what that does to the "
              "report.")

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
             f"relevance={args.relevance} tuning={args.tuning} "
             + (f"setting={setting_text(model.tuned_['params'])} "
                if tuned else "") +
             f"stopping={args.stopping} trees={best} test_rows={len(test)} "
             f"positive_rate={floor:.4f} "
             f"best_baseline={best_base}:{bb:.4f} "
             f"holdout_from={HOLDOUT_START.date()} "
             f"test_changes={n_changes} test_upgrades={n_upgrades} "
             f"pr_auc_95={ci['pr_auc'][0]:.4f}-{ci['pr_auc'][1]:.4f} "
             f"lift_vs_popularity={lift_pop:.2f}x "
             f"lift_95={ci['lift'][0]:.2f}-{ci['lift'][1]:.2f} "
             f"linear={lin:.4f} lift_vs_linear={lift_lin:.2f}x "
             f"lift_vs_linear_95={ci_lin['lift'][0]:.2f}-"
             f"{ci_lin['lift'][1]:.2f}")

    # CARRY THE RANGE INTO THE ROW ITSELF. pr_auc here is ONE cut date, and
    # §5.6 measured that a single cut can sit anywhere in a band half as
    # wide as the number itself — this run's 0.3924 is near the top of
    # 0.309–0.438. The database is read by a website that will print
    # whatever it finds, and a context-free metric on a public page is the
    # exact failure this project spent two days documenting. `notes` is
    # free text and reaches the API unchanged, so the range rides along
    # with the number instead of living only in a notebook.
    stab = DATA / stability_name(label, args.stopping, args.objective,
                                 args.relevance, tuning=args.tuning)
    rerun = (f"ml/model/stability.py --label {label} --stopping "
             f"{args.stopping}"
             + (f" --objective {args.objective}"
                if args.objective != "lambdarank" else "")
             + (" --relevance graded" if graded else "")
             + (" --tuning cv" if tuned else ""))
    # A stability file written BEFORE the holdout froze scored every cut on
    # test halves that ran to the end of the data, holdout rows included.
    # Quoting it here would carry pre-freeze numbers into model_run and on
    # to the site. stability.py now stamps each row with the boundary it
    # ran under; a file without the stamp, or with a different one, is
    # stale and is not quoted.
    #
    # The same goes for what the sweep's models were taught. A range
    # measured on binary models says nothing about a graded one, and the
    # reverse, so each row also says its relevance (a file written before
    # item 3 has none, and was binary) and a mismatch is not quoted either.
    fresh = False
    if not stab.exists():
        print(f"note: {stab} not found, so this model_run row will carry a "
              f"single-cut\nnumber with no range. Run {rerun} first.")
    else:
        st = pd.read_csv(stab)
        fresh = ("holdout_from" in st and len(st) > 0 and
                 (st["holdout_from"].astype(str)
                  == str(HOLDOUT_START.date())).all())
        taught = (st["relevance"].astype(str) if "relevance" in st
                  else pd.Series("binary", index=st.index))
        if not fresh:
            print(f"note: {stab} predates the holdout freeze, so its cuts "
                  "included holdout rows.\nNot quoted. Re-run "
                  f"{rerun}, then this script.")
        elif not taught.eq(args.relevance).all():
            fresh = False
            print(f"note: {stab} was measured on models with "
                  f"{', '.join(sorted(set(taught)))} relevance, and this one "
                  f"is {args.relevance}.\nNot quoted. Re-run {rerun}, then "
                  "this script.")
        # And how its trees were sized (F7). A file from before item 7 has
        # no stamp and was fixed.
        sized = (st["tuning"].astype(str) if "tuning" in st
                 else pd.Series("fixed", index=st.index))
        if fresh and not sized.eq(args.tuning).all():
            fresh = False
            kinds = ", ".join(sorted(set(sized)))
            print(f"note: {stab} was measured on {kinds}-tuned models, and "
                  f"this one is {args.tuning}.\nNot quoted. Re-run {rerun}, "
                  "then this script.")
    if fresh:
        st = st[~st["skipped"].astype(bool)] if "skipped" in st else st
        if not st.empty:
            lift = st["lift_vs_pop"].astype(float)
            wins = int(st["beats_pop"].astype(bool).sum())
            # "popularity", not best_base: both columns in the stability
            # file are measured against popularity. Naming whichever
            # baseline won THIS split put a claim nobody measured into a
            # row the site prints, as soon as kind_prior won one.
            notes += (f" | across {len(st)} cut dates: beats popularity "
                      f"{wins}/{len(st)}, lift vs popularity median "
                      f"{lift.median():.2f}x min {lift.min():.2f}x "
                      f"max {lift.max():.2f}x")
            # A sweep from before item 5 has no intervals; say nothing
            # about them rather than read their absence as zero.
            if "lift_lo" in st and st["lift_lo"].notna().all():
                clear = int((st["lift_lo"].astype(float) > 1).sum())
                notes += (f", lift's 95% interval above 1.0x at "
                          f"{clear}/{len(st)}")
        else:
            print("note: stability file has no usable splits; the model_run "
                  "row will carry a single-cut number with no range.")

    # positive_rate IS A TOP-LEVEL FIELD, not just a phrase inside notes.
    # It has been in the notes string since day 5 — recoverable only by
    # parsing prose, which is no way to store half a result. PR-AUC's floor
    # IS the positive rate, so a stored pr_auc without it cannot be read by
    # anyone who was not in the room. Migration 005 gave it a column and
    # the API contract (decision 13) forbids showing one without the other,
    # so this key is what that rule reads.
    run = {"version": version, **{k: round(v, 6) for k, v in m.items()},
           "positive_rate": round(floor, 6), "notes": notes}
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
