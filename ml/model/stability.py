"""
Does the result survive more than one arbitrary date?

    python ml/model/stability.py
    python ml/model/stability.py --label label_alias
    python ml/model/stability.py --all-labels
    python ml/model/stability.py --label label --at 2025-09-29,2026-01-13
    python ml/model/stability.py --relevance graded
    python ml/model/stability.py --objective binary
    python ml/model/stability.py --tuning cv

Reads  data/features.csv
Writes data/stability_<label>_<stopping>.csv, or with --at
       data/stability_<label>_<stopping>_at.csv, which train.py quotes
       only for path length's row, and only cut at the seven reference
       dates (NOTES §36.3).
       Another objective adds its name, then _graded, _tuned, _at:
       data/stability_label_alias_cv_binary.csv,
       data/stability_label_alias_cv_graded.csv,
       data/stability_label_alias_cv_tuned.csv (train.stability_name).
       Every row says the relevance and the tuning it was trained with,
       and train.py quotes a file only for a model trained the same way.
       A tuned row also says the setting its cut chose (`setting`).

Each row also carries ndcg_20_graded: nDCG@20 over the same upgrades,
with a change's gain 1, 3, 7 or 15 by how many packages use it (F2). It is
measured whatever the model was taught, so a binary sweep and a graded
one can be compared on it (scripts/item3_relevance.py).

And, since item 7 (F8), the line: `linear` is the PR-AUC at that cut of a
logistic regression on the model's own features, fitted on the same rows
the model was, with `linear_ndcg_20`, `lift_vs_linear` (model PR-AUC over
the line's) and `beats_linear`. scripts/item7_linear.py reads a
lambdarank sweep and a binary one and applies NOTES §29.1's rule.

And, since 10 Oct, two things beside each other (metrics.tie_averaged).
F42 (NOTES §35): `pr_auc_ties` and `popularity_ties` are the model's and
popularity's PR-AUC with ties averaged, with `lift_vs_pop_ties`, its
interval `lift_lo_ties`-`lift_hi_ties`, and `beats_pop_ties`; every other
number kept sklearn's convention until the report moved, on 11 Oct, and
since then they repeat the plain columns (below). F41 (NOTES
§34): `path` is the PR-AUC, ties averaged, of ranking by the shortest
import path (minus public_depth, nothing fitted), with `path_p_at_10`,
`path_ndcg_20`, `lift_vs_path` (pr_auc_ties over path, to four places)
and its interval `lift_vs_path_lo`-`lift_vs_path_hi`, `beats_path` and
`beats_path_ndcg`. scripts/path_verdict.py reads the sweep at the seven
reference dates and applies §34.1's rule; every row now also says its
label and objective, which the verdict checks.

And, from 10 Oct night, item 15 (F25, NOTES §36): which ranker ships.
In a sweep of the shipped model (label_alias, lambdarank, binary, tuned
by CV), each cut also fits R, the two reachability features
(public_depth, has_export_path) chosen and fitted exactly as the model
is, and scores two orders that break ties with the model: path length,
then the model (`path_model`), and R, then the model (`reach_model`).
Every one of the five rankers gets PR-AUC with ties averaged and, under
`_e` names, precision@10 and nDCG@20 averaged over every order of its
ties (F44); R's size, setting and number of distinct scores, a
fingerprint of the test half and what the sweep ran on are written
beside them (CONTEST, CONTEST_EXTRA). None of it is printed here:
scripts/ship_verdict.py reads the sweep at the seven reference dates,
checks it, and applies §36.1's rule.

And, from 11 Oct (NOTES §36.2): path length alone ships, so two things
change. Every number takes one convention, metrics.CONVENTION: PR-AUC
with ties averaged and precision@10 and nDCG@20 averaged over every
order of the ties, and every row says so (`convention`). A file without
the stamp was measured the old way, and train.py and compare_sweeps.py
do not set it beside one with it. Its F42 columns (`pr_auc_ties` and the
rest) now repeat the plain ones and are kept so older readers still
work. And each row carries path length's own lift over popularity, the
number the report quotes: `path_lift_vs_pop`, its paired interval
`path_lift_lo`-`path_lift_hi`, and `path_beats_pop`. Item 15's columns
are written only when asked for (--contest); its verdict is in.

BEFORE AND AFTER A FIX, AT THE SAME DATES (--at). The seven cut dates are
quantiles, so they move whenever the rows do: item 2 deleted 708 dev
upgrades and every date moved. A fix judged by comparing a sweep before
it with a sweep after it would then be judged on two different sets of
windows, which is §15's error (two models given different exams). --at
takes the dates the "before" sweep printed and cuts there instead, so
the "after" is measured on the same windows. With released_at written as
plain dates, cutting at a printed date reproduces that cut's split exactly.

WHY THIS IS NOTES ITEM 0 AND NOT A NICE-TO-HAVE.

Every number this project has reported comes from ONE cut date. Everything
older trains, everything newer tests. That single choice decides the test
positive rate, which decides PR-AUC's floor, which decides whether a lift
looks like 1.8x or 3.1x. It also decides the validation slice, which
decides where early stopping halts — measured anywhere between 1 and 120
trees across nine ablation runs on the same data.

That is not a robustness footnote. It produced a WRONG CLAIM. NOTES §5.3
reported "path shape alone reaches 105% of the full model, therefore the
model IS the depth heuristic". Re-running with consistent feature groups
showed every subset beating the full model under that label — dropping
popularity alone doubled PR-AUC. A 15-feature model losing to every subset
of itself is an unstable evaluation, not a hidden heuristic. The 105% was
variance, read as a finding, and it sat in the notebook for days.

So: cut at SEVERAL dates, refit at each, and report the spread. A result
that only holds at one cut is not a result, it is a coincidence with a
decimal point.

WHAT THIS DOES NOT DO. These splits are not independent — they share most
of their training data and overlap heavily in test — so the spread is a
sensitivity range, not a confidence interval, and it must never be written
with a +/- that implies statistics it does not have. Report it as "median,
and the range across N cut dates". That is an honest claim and a useful
one: it says how much the answer moves when the arbitrary choice moves.

THE BASELINES ARE REFITTED AT EVERY CUT TOO. kind_prior is fitted on train
only, and popularity's PR-AUC depends on the test half's composition, so
carrying one cut's baseline across all of them would compare a moving
model against a fixed target and flatter whichever way the data leaned.
Lift is computed WITHIN each split, then summarised.
"""

import argparse
import hashlib
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import HOLDOUT_START, assert_no_holdout  # noqa: E402
from ml.model.ablate import REACHABILITY  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import (CONVENTION, evaluate,  # noqa: E402
                              intervals, n_rankable, ndcg_at_k,
                              tie_averaged)
from ml.model.train import (COUNT_OF, FIXED_PARAMS, GROUP,  # noqa: E402
                            RELEVANCE, SHIPPED_RELEVANCE, SHIPPED_TUNING,
                            TUNING, fit_cv, fit_model, graded_gain,
                            prepare, relevance_problem, score_with,
                            setting_text, split_valid, stability_name,
                            tuning_problem)

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"

# Where to cut. Spread across the middle of the date range: earlier than
# 0.55 leaves too little training data, later than 0.85 leaves a test half
# with almost no positives, and both ends produce numbers that say more
# about the cut than about the model.
#
# QUANTILES OF UPGRADES, NOT ROWS (F38, NOTES §22.5). Each version pair has
# one date, and cut() takes the quantile over those dates, so a release
# counts once however many changes it holds. Taken over rows, the cut
# points followed the biggest releases: on the 26 Sep dev set, 5 Apr held
# 2,296 rows and 21 Jan 1,313, a fifth of dev between them, and those two
# days swallowed five of the seven cut points, leaving 4 distinct splits.
# Decided on that reasoning and written into NOTES before it was first run.
CUTS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

MIN_TEST_POSITIVES = 30

# A cut with four rankable version pairs is not a weak measurement of
# precision@10. It is not a measurement of precision@10.
#
# A pair is rankable at 10 only if it has more than 10 changes AND at least
# one positive — otherwise ordering it is either impossible or trivially
# correct. The q=0.85 cut leaves 4 or 5 such pairs, and its lift was the
# WORST CASE for both candidate labels, which is to say the ship decision
# was resting on the least measurable cut in the sweep.
#
# ADDED 14 SEP, AFTER SEEING IT MATTER — and that has to be said out loud,
# because choosing a rule once you can see who it favours is the same
# family of error as picking features on test PR-AUC (§5.4).
#
# Two things defend it. The justification never references which label
# wins: four pairs cannot support a top-10 metric no matter whose model is
# being scored, and MIN_TEST_POSITIVES was already conceding that tiny
# cuts are invalid — it just gated on the wrong quantity. And it is
# checkable against data collected before the rule existed:
#
#     lift_MIN          with q=0.85        without it
#     12 Sep dataset    alias  1.94 win    alias  2.19 win
#     14 Sep dataset    scoped 1.92 win    alias  2.32 win
#
# The ungated rule reversed when 50 rows (0.26% of the data) were added,
# because the cut dates are QUANTILES and every one of them moves when the
# row count does. The gated rule gives the same answer on both datasets.
# A decision rule that flips on a quarter of a percent of the data is not
# measuring the thing it claims to measure.
MIN_RANKABLE_PAIRS = 10

# Item 15's five rankers (NOTES §36), simplest first, each as its columns
# in a sweep row: PR-AUC with ties averaged (F42), and nDCG@20 and
# precision@10 averaged over every order of the ties (`_e`, F44).
# scripts/ship_verdict.py reads the same columns.
CONTEST = {
    "path length alone": ("path", "path_ndcg_20_e", "path_p_at_10_e"),
    "the two reachability features": ("reach", "reach_ndcg_20_e",
                                      "reach_p_at_10_e"),
    "the model": ("pr_auc_ties", "ndcg_20_e", "p_at_10_e"),
    "path length, then the model": ("path_model", "path_model_ndcg_20_e",
                                    "path_model_p_at_10_e"),
    "the two features, then the model": ("reach_model",
                                         "reach_model_ndcg_20_e",
                                         "reach_model_p_at_10_e"),
}
# Written beside them: R's size, setting and how many distinct scores it
# gives, a fingerprint of the test half it was all measured on, and what
# it ran on, so a sweep that does not reproduce §34.2 can say why.
CONTEST_EXTRA = ("reach_trees", "reach_setting", "reach_levels", "test_hash",
                 "lightgbm", "sklearn", "pandas", "numpy", "python", "cpus",
                 "omp_threads", "commit")


def upgrade_dates(df: pd.DataFrame) -> pd.Series:
    """One date per version pair, the newest of its rows. On clean data
    every row of a pair already carries version_to's date."""
    when = pd.to_datetime(df["released_at"], errors="coerce")
    return when.groupby([df[c] for c in GROUP]).max()


def cut(df: pd.DataFrame, point) -> tuple[pd.Timestamp, pd.Series]:
    """The cut date for one cut point, and which rows it sends to test.

    A number is a quantile of the upgrades (see CUTS for why upgrades and
    not rows). A 'YYYY-MM-DD' string is that date itself (--at): rows
    dated on or before it train, rows after it test.

    Undated rows train, never test — the same rule as build.temporal_split.
    """
    if isinstance(point, str):
        cutoff = pd.Timestamp(point)
    else:
        cutoff = upgrade_dates(df).dropna().quantile(point)
    when = pd.to_datetime(df["released_at"], errors="coerce")
    return cutoff, (when > cutoff).fillna(False).astype(bool)


def qval(point) -> float:
    """The quantile a cut point stands for; NaN for a fixed date."""
    return float("nan") if isinstance(point, str) else point


def show(point) -> str:
    """How a cut point is named in the report."""
    return point if isinstance(point, str) else f"q={point:.2f}"


def one_split(df: pd.DataFrame, q, label: str, feats: list[str],
              objective: str, stopping: str = "cv",
              relevance: str = "binary",
              tuning: str = "fixed", contest: bool = False) -> dict | None:
    """Train and score at one cut point (a quantile, or a date with --at).
    None if the split is unusable.

    The gates below read the 0/1 label whatever the relevance, so a
    binary sweep and a graded one skip exactly the same cuts.

    contest=True adds item 15's rankers (NOTES §36): R, fitted after the
    model and exactly as it was, on the two reachability features, and the
    two orders that break ties with the model. Off by default, so the
    scripts that re-measure earlier results fit what they always did; the
    model's own numbers are the same either way."""
    cutoff, is_test = cut(df, q)
    test = df[is_test].sort_values(GROUP)
    full_train = df[~is_test].sort_values(GROUP)

    pos = int(test[label].sum())
    # Checked BEFORE fitting — an unusable cut should not cost a model.
    rankable = n_rankable(test, label, 10)
    if (pos < MIN_TEST_POSITIVES or rankable < MIN_RANKABLE_PAIRS
            or full_train.empty):
        return {"cut": str(cutoff.date()), "q": qval(q),
                "test_rows": len(test), "test_pos": pos,
                "rankable10": rankable, "skipped": True}

    train, valid, _ = split_valid(full_train)
    train, valid = train.sort_values(GROUP), valid.sort_values(GROUP)
    if train.empty or valid.empty or train[label].sum() == 0:
        return {"cut": str(cutoff.date()), "q": qval(q),
                "test_rows": len(test), "test_pos": pos,
                "rankable10": rankable, "skipped": True}

    if stopping == "cv":
        model, trees, _folds = fit_cv(full_train, feats, label, objective,
                                      relevance=relevance, tuning=tuning)
        fit_on = full_train
    else:
        model = fit_model(train, valid, feats, label, objective,
                          relevance=relevance)
        trees = getattr(model, "best_iteration_", None) or 600
        fit_on = train

    scored = test.copy()
    scored["model"] = score_with(model, test, feats)
    scored = add_baseline_scores(fit_on, scored, label)

    m = evaluate(scored, "model", label)
    pop = evaluate(scored, "popularity", label)
    sem = evaluate(scored, "semver", label)
    # F8: the line, fitted on the same rows this cut's model was.
    lin = evaluate(scored, "linear", label)
    floor = float(test[label].mean())
    # Measured for binary and graded models alike; NaN only where the file
    # has no count to grade by.
    ndcg_graded = (ndcg_at_k(scored.assign(_gain=graded_gain(scored, label)),
                             "model", label, 20, gain="_gain")
                   if COUNT_OF.get(label) in scored else float("nan"))
    # F11: this cut's own uncertainty, over its test half's upgrades with
    # the model held fixed. The spread ACROSS cuts, which report() prints
    # as the range, is the other kind; neither stands in for the other.
    ci = intervals(scored, label, "model", "popularity")
    # F42 and F41 (NOTES §35, §34): the tie-averaged PR-AUCs and their
    # lifts' intervals, and path length alone. Until 11 Oct these sat
    # beside the old convention's numbers; since metrics.CONVENTION every
    # number above is tie-averaged too, and the F42 columns repeat them.
    tie = tie_averaged(scored, label)

    # Item 15 (NOTES §36), after everything above is measured, so the
    # model's numbers cannot depend on it.
    extra = (contest_columns(scored, label, full_train, train, valid,
                             objective, stopping, relevance, tuning)
             if contest else {})

    return {
        "cut": str(cutoff.date()),
        "q": qval(q),
        "test_rows": len(test),
        "test_pos": pos,
        "floor": round(floor, 4),
        "trees": trees,
        "rankable10": n_rankable(test, label, 10),
        "pr_auc": round(m["pr_auc"], 4),
        "p_at_10": round(m["precision_at_10"], 4),
        "ndcg_20": round(m["ndcg_at_20"], 4),
        "ndcg_20_graded": round(ndcg_graded, 4),
        "popularity": round(pop["pr_auc"], 4),
        "semver": round(sem["pr_auc"], 4),
        "lift_vs_pop": round(m["pr_auc"] / pop["pr_auc"], 2)
        if pop["pr_auc"] else float("nan"),
        "lift_lo": round(ci["lift"][0], 2),
        "lift_hi": round(ci["lift"][1], 2),
        "pr_auc_lo": round(ci["pr_auc"][0], 4),
        "pr_auc_hi": round(ci["pr_auc"][1], 4),
        "test_changes": len(test.drop_duplicates(GROUP + ["symbol"])),
        "test_upgrades": ci["upgrades"],
        "beats_pop": bool(m["pr_auc"] > pop["pr_auc"]),
        "beats_semver": bool(m["pr_auc"] > sem["pr_auc"]),
        **against_the_line(m, lin),
        **with_ties(tie),
        **against_path(m, tie),
        **path_over_popularity(tie),
        # F7: the setting this cut's model was fitted with, as
        # leaves/learning rate/smallest leaf. Always 31/0.05/30 untuned.
        "setting": setting_text(getattr(model, "tuned_", {}).get(
            "params", FIXED_PARAMS)),
        **extra,
        "skipped": False,
    }


def then_by(primary, secondary) -> np.ndarray:
    """A score that orders rows by `primary`, and the rows `primary` ties
    by `secondary`: for item 15 (NOTES §36), shortest path first, and among
    changes equally short, the model's order. Rows tied on both stay tied.

    primary's distinct values become their dense ranks, 1 apart whatever
    the gaps between them (R's scores can differ in the fifth decimal), and
    secondary adds less than 0.5 by its rank, so it can reorder only inside
    a tie. Ranks are taken over the whole frame, which keeps every
    upgrade's own order as well."""
    p = pd.Series(np.asarray(primary, dtype=float))
    s = pd.Series(np.asarray(secondary, dtype=float))
    if p.isna().any() or s.isna().any():
        raise ValueError("then_by needs a score on every row")
    return (p.rank(method="dense").to_numpy()
            + 0.5 * s.rank(method="average").to_numpy() / (len(s) + 1))


def contest_columns(scored: pd.DataFrame, label: str,
                    full_train: pd.DataFrame, train: pd.DataFrame,
                    valid: pd.DataFrame, objective: str, stopping: str,
                    relevance: str, tuning: str) -> dict:
    """Item 15's columns of a sweep row (NOTES §36.1), from a test half the
    model and the baselines have already scored.

    R is the two reachability features, fitted by exactly the procedure
    the model was (one_split's): with stopping cv, its setting and tree
    count chosen on the same folds (tuning cv) and a refit on all of
    train. `path_model` and `reach_model` are path length's order and R's,
    their ties broken by the model's (then_by). Every one of the five is
    scored the same way: PR-AUC with ties averaged (F42; for path and the
    model, the `path` and `pr_auc_ties` columns already written), and
    precision@10 and nDCG@20 averaged over every order of whatever still
    ties (F44, the `_e` columns)."""
    if stopping == "cv":
        reach, trees, _ = fit_cv(full_train, REACHABILITY, label, objective,
                                 relevance=relevance, tuning=tuning)
    else:
        reach = fit_model(train, valid, REACHABILITY, label, objective,
                          relevance=relevance)
        trees = getattr(reach, "best_iteration_", None) or 600
    test = scored.copy()
    test["reach"] = score_with(reach, test, REACHABILITY)
    test["path_model"] = then_by(test["path"], test["model"])
    test["reach_model"] = then_by(test["reach"], test["model"])
    out = {"reach_trees": trees,
           "reach_setting": setting_text(getattr(reach, "tuned_", {}).get(
               "params", FIXED_PARAMS)),
           "reach_levels": int(test["reach"].nunique())}
    for name in ("path", "reach", "model", "path_model", "reach_model"):
        e = evaluate(test, name, label, ties=True, expected=True)
        if name in ("reach", "path_model", "reach_model"):
            out[name] = round(e["pr_auc"], 4)
        prefix = "" if name == "model" else f"{name}_"
        out[f"{prefix}ndcg_20_e"] = round(e["ndcg_at_20"], 4)
        out[f"{prefix}p_at_10_e"] = round(e["precision_at_10"], 4)
    out["test_hash"] = test_hash(test, label)
    return {**out, **environment()}


def _cell(v) -> str:
    """One value as test_hash writes it, the same on every pandas: a
    missing value empty, a number by its value (2, 2.0 and True alike as
    2 and 1), anything else as text."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return ""
    if isinstance(v, (bool, np.bool_, int, np.integer, float, np.floating)):
        return f"{float(v):g}"
    return str(v)


def test_hash(test: pd.DataFrame, label: str) -> str:
    """A fingerprint of a test half: its changes, their label and the two
    reachability features, each value written by _cell and the rows in a
    fixed order. Two sweeps measured on the same rows have the same one,
    whichever pandas wrote them."""
    cols = [c for c in GROUP + ["symbol", "sub_target", label,
                                "public_depth", "has_export_path"]
            if c in test]
    rows = sorted("\x1f".join(_cell(v) for v in r)
                  for r in test[cols].astype(object).itertuples(index=False))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:12]


def environment() -> dict:
    """What a sweep ran on: library versions, CPUs, the OpenMP thread
    setting LightGBM reads, and the commit, marked -dirty if the working
    tree had changes. A refit can come out differently on another machine
    or another LightGBM, so a sweep that does not reproduce an earlier one
    says what changed."""
    import platform

    import lightgbm
    import sklearn
    here = pathlib.Path(__file__).resolve().parents[2]

    def git(*args) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True,
                                  text=True, timeout=10,
                                  cwd=here).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = git("rev-parse", "--short", "HEAD")
    if commit and git("status", "--porcelain", "--untracked-files=no"):
        commit += "-dirty"
    return {"lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "cpus": os.cpu_count(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "commit": commit or "unknown"}


def against_the_line(m: dict, lin: dict) -> dict:
    """The F8 columns of a sweep row, from the model's and the line's
    evaluate() dicts: the line's PR-AUC and nDCG@20, the model's PR-AUC
    over the line's, and whether the model is ahead. A tie is not ahead."""
    return {
        "linear": round(lin["pr_auc"], 4),
        "linear_ndcg_20": round(lin["ndcg_at_20"], 4),
        "lift_vs_linear": round(m["pr_auc"] / lin["pr_auc"], 2)
        if lin["pr_auc"] else float("nan"),
        "beats_linear": bool(m["pr_auc"] > lin["pr_auc"]),
    }


def with_ties(tie: dict) -> dict:
    """The F42 columns of a sweep row (NOTES §35), from tie_averaged(): the
    model's and popularity's PR-AUC with ties averaged, the lift between
    them with its 95% interval, and whether the model is ahead. Until 11
    Oct they sat beside the old convention's numbers; now (NOTES §36.3)
    they repeat the plain columns, and are kept for older readers."""
    return {
        "pr_auc_ties": round(tie["model"], 4),
        "popularity_ties": round(tie["popularity"], 4),
        "lift_vs_pop_ties": round(tie["lift_pop"], 2),
        "lift_lo_ties": round(tie["ci_pop"][0], 2),
        "lift_hi_ties": round(tie["ci_pop"][1], 2),
        "beats_pop_ties": bool(tie["model"] > tie["popularity"]),
    }


def against_path(m: dict, tie: dict) -> dict:
    """The F41 columns of a sweep row (NOTES §34.1), from the model's
    evaluate() dict and tie_averaged(): path's PR-AUC (ties averaged,
    F42), precision@10 and nDCG@20; the model's PR-AUC, ties averaged too,
    over path's, with that lift's 95% interval; and whether the model is
    ahead on PR-AUC and on nDCG@20. Ratio and flags come from the values
    as stored, to four places, which is what scripts/path_verdict.py
    reads, so a tie there is ahead for neither."""
    pth = tie["path"]
    mp, pp = round(tie["model"], 4), round(pth["pr_auc"], 4)
    mn, pn = round(m["ndcg_at_20"], 4), round(pth["ndcg_at_20"], 4)
    return {
        "path": pp,
        "path_p_at_10": round(pth["precision_at_10"], 4),
        "path_ndcg_20": pn,
        "lift_vs_path": round(mp / pp, 4) if pp else float("nan"),
        "lift_vs_path_lo": round(tie["ci_path"][0], 4),
        "lift_vs_path_hi": round(tie["ci_path"][1], 4),
        "beats_path": bool(mp > pp),
        "beats_path_ndcg": bool(mn > pn),
    }


def path_over_popularity(tie: dict) -> dict:
    """Path length's own columns as the ranker that ships (NOTES §36.2),
    from tie_averaged(): its lift over popularity, both PR-AUCs with ties
    averaged, the paired 95% interval of that lift, and whether it is
    ahead. Worked out as the model's are: the lift to two places from the
    unrounded PR-AUCs, and a tie is not ahead."""
    pth, pop = tie["path"]["pr_auc"], tie["popularity"]
    return {
        "path_lift_vs_pop": round(pth / pop, 2) if pop else float("nan"),
        "path_lift_lo": round(tie["ci_path_pop"][0], 2),
        "path_lift_hi": round(tie["ci_path_pop"][1], 2),
        "path_beats_pop": bool(pth > pop),
    }


def run_label(df: pd.DataFrame, label: str, feats: list[str],
              objective: str, stopping: str = "cv",
              points: list | None = None,
              relevance: str = "binary",
              tuning: str = "fixed", contest: bool = False) -> pd.DataFrame:
    """One row per cut point: CUTS, or the dates given with --at. contest
    adds item 15's rankers to every row (one_split)."""
    # ONE SPLIT, ONE VOTE.
    #
    # When a single day holds more upgrades than lie between two
    # neighbouring cut points, both quantiles land on that day and produce
    # the same split, row for row: the same train half, the same test half,
    # the same model and the same numbers. Found 26 Sep, in the first sweep
    # after the freeze, while the cut points were still quantiles of ROWS:
    # q=0.75 and q=0.80 both cut at 2026-04-05 with the same 2,902 test
    # rows, so "beats popularity at 7/7" and the 2.01x median counted one
    # measurement twice. Six splits had been measured. F38 moved the
    # quantiles to upgrades (see CUTS), which makes a repeat rare; this is
    # what makes one harmless when it still happens.
    #
    # Counting a split twice is wrong whichever way it moves the median, so
    # the repeat stays in the file as a skipped row that names the cut it
    # repeats, and nothing downstream counts it: not the report, not
    # train.py's notes, not the label-vs-label table. Identity is checked on
    # the rows themselves, not the printed date, because two cut points can
    # differ in time of day and still select the same rows.
    first: dict[bytes, object] = {}
    rows = []
    for q in (points or CUTS):
        cutoff, is_test = cut(df, q)
        key = np.asarray(is_test, dtype=bool).tobytes()
        if key in first:
            test = df[is_test]
            # The day both quantiles fell on: the newest date left in train.
            when = pd.to_datetime(df["released_at"], errors="coerce")
            day = when[~is_test].max().normalize()
            on_day = upgrade_dates(df).dt.normalize().eq(day)
            rows.append({"cut": str(cutoff.date()), "q": qval(q),
                         "test_rows": len(test),
                         "test_pos": int(test[label].sum()),
                         "rankable10": n_rankable(test, label, 10),
                         "skipped": True, "same_as": first[key],
                         "shared_day": str(day.date()),
                         "shared_day_upgrades": int(on_day.sum())})
            continue
        first[key] = q
        rows.append(one_split(df, q, label, feats, objective, stopping,
                              relevance=relevance, tuning=tuning,
                              contest=contest))
    t = pd.DataFrame([r for r in rows if r])
    # Stamp the boundary this sweep ran under. Every file written before the
    # freeze lacks the column, and its later cuts scored holdout rows, so
    # train.py refuses to quote any file whose stamp is missing or different.
    t["holdout_from"] = str(HOLDOUT_START.date())
    # And what its models were taught. train.py quotes a sweep only for a
    # model trained the same way; a file from before item 3 has no stamp
    # and was binary.
    t["relevance"] = relevance
    # And how its trees were sized (F7); a file from before item 7 has no
    # stamp and was fixed.
    t["tuning"] = tuning
    # And, from 10 Oct, which label and objective: a sweep's file name says
    # them, and scripts/path_verdict.py checks the rows too (F41).
    t["label"] = label
    t["objective"] = objective
    # And, from 11 Oct, how ties were counted (NOTES §36.2): a file without
    # the stamp took scikit-learn's PR-AUC and one fixed order of the ties,
    # and is never set beside one with it.
    t["convention"] = CONVENTION
    return t


def report(t: pd.DataFrame, label: str) -> None:
    repeat = (t["same_as"].notna() if "same_as" in t
              else pd.Series(False, index=t.index))
    print("\n" + "=" * 74)
    print(f"  {label}   —   {len(t)} cut points, "
          f"{int((~repeat).sum())} distinct splits")
    print("=" * 74)

    skipped = t[t["skipped"]]
    t = t[~t["skipped"]].copy()
    if skipped.shape[0]:
        for i, r in skipped.iterrows():
            where = (f"q={r.q:.2f} ({r.cut})" if pd.notna(r.q) else r.cut)
            if repeat.loc[i]:
                print(f"  skipped {where}: the same split as "
                      f"{show(r.same_as)}, row for row, so not a second "
                      f"measurement\n      (both cut points fall among the "
                      f"{int(r.shared_day_upgrades):,} upgrades dated "
                      f"{r.shared_day})")
                continue
            why = []
            if int(r.test_pos) < MIN_TEST_POSITIVES:
                why.append(f"{int(r.test_pos)} test positives "
                           f"(min {MIN_TEST_POSITIVES})")
            rk = r.get("rankable10")
            if pd.notna(rk) and int(rk) < MIN_RANKABLE_PAIRS:
                why.append(f"{int(rk)} rankable pairs "
                           f"(min {MIN_RANKABLE_PAIRS})")
            print(f"  skipped {where}: "
                  + ", ".join(why or ["no usable train/valid split"]))
    if t.empty:
        print("  no usable splits.")
        return

    cols = ["cut", "test_rows", "test_pos", "floor", "trees", "rankable10",
            "pr_auc", "popularity", "lift_vs_pop", "p_at_10", "ndcg_20"]
    # Each cut's lift with its own 95% interval (F11), where the file has
    # them; a sweep from before item 5 does not.
    has_ci = "lift_lo" in t and t["lift_lo"].notna().all()
    if has_ci:
        t["lift_95"] = [f"{lo:.2f}-{hi:.2f}"
                        for lo, hi in zip(t["lift_lo"], t["lift_hi"])]
        cols.insert(cols.index("lift_vs_pop") + 1, "lift_95")
    if "ndcg_20_graded" in t and t["ndcg_20_graded"].notna().any():
        cols.append("ndcg_20_graded")
    # F7: the setting each cut chose, when the sweep was tuned.
    if "tuning" in t and t["tuning"].astype(str).eq("cv").all() \
            and "setting" in t:
        cols.insert(cols.index("trees") + 1, "setting")
    # F8: the line beside the model at each cut, where the file has it; a
    # sweep from before item 7 does not.
    has_line = "linear" in t and t["linear"].notna().all()
    if has_line:
        cols.insert(cols.index("popularity"), "linear")
        cols.append("linear_ndcg_20")
    # F42 and F41: PR-AUC with ties averaged, and path length alone,
    # where the file has them. In a sweep on CONVENTION every number is
    # tie-averaged already and the F42 columns repeat the plain ones, so
    # they are not shown twice.
    same = ("convention" in t
            and t["convention"].astype(str).eq(CONVENTION).all())
    has_ties = (not same and "pr_auc_ties" in t
                and t["pr_auc_ties"].notna().all())
    if has_ties:
        cols.insert(cols.index("pr_auc") + 1, "pr_auc_ties")
        cols.insert(cols.index("popularity") + 1, "popularity_ties")
        after = "lift_95" if "lift_95" in cols else "lift_vs_pop"
        cols.insert(cols.index(after) + 1, "lift_vs_pop_ties")
    has_path = "path" in t and t["path"].notna().all()
    if has_path:
        cols.insert(cols.index("popularity"), "path")
        cols.append("path_ndcg_20")
    # Path length's own lift over popularity, with its interval, where the
    # file has it (NOTES §36.2): the ranker that ships.
    has_ship = (has_path and "path_lift_vs_pop" in t
                and t["path_lift_vs_pop"].notna().all())
    if has_ship:
        t["path_lift_95"] = [f"{lo:.2f}-{hi:.2f}" for lo, hi in
                             zip(t["path_lift_lo"], t["path_lift_hi"])]
        at = cols.index("path") + 1
        cols[at:at] = ["path_lift_vs_pop", "path_lift_95"]
    # A skipped row has no tree count, which turns the column into floats
    # ("20.0") for the rows that do. Every fitted row has a whole number.
    t["trees"] = t["trees"].astype(int)
    print()
    print(t[cols].to_string(index=False))

    def band(col: str) -> str:
        v = t[col].astype(float)
        return (f"{v.median():.4f}   range {v.min():.4f} – {v.max():.4f}"
                f"   spread {v.max() - v.min():.4f}")

    print(f"\n  PR-AUC        {band('pr_auc')}")
    print(f"  floor         {band('floor')}")
    print(f"  lift vs pop   {t['lift_vs_pop'].median():.2f}x   "
          f"range {t['lift_vs_pop'].min():.2f}x – {t['lift_vs_pop'].max():.2f}x")
    print(f"  trees         {int(t['trees'].min())} – {int(t['trees'].max())}")

    wins = int(t["beats_pop"].sum())
    gate = int(t["beats_semver"].sum())
    n = len(t)
    print(f"\n  beats popularity at {wins}/{n} cut dates")
    if has_ci:
        clear = int((t["lift_lo"].astype(float) > 1).sum())
        print(f"  and by more than its own 95% interval (lower end above "
              f"1.0x) at {clear}/{n}")
    print(f"  beats semver (the kill-date gate) at {gate}/{n}")
    if has_line:
        # Point comparisons on one test half, so the ratio is on one scale
        # at each cut; across cuts the floors differ, so the median of the
        # ratios, not of the differences. NOTES §29.1 says how to read it.
        ratio = t["lift_vs_linear"].astype(float)
        print(f"  beats the line (logistic regression, same features) at "
              f"{int(t['beats_linear'].sum())}/{n}; model/line PR-AUC "
              f"median {ratio.median():.2f}x, range {ratio.min():.2f}x – "
              f"{ratio.max():.2f}x")
    if has_ties:
        # F42 (NOTES §35): the same lift with every PR-AUC's ties averaged.
        lt = t["lift_vs_pop_ties"].astype(float)
        print(f"  with ties averaged (F42): lift vs pop median "
              f"{lt.median():.2f}x, range {lt.min():.2f}x – {lt.max():.2f}x;"
              f" beats popularity\n  at {int(t['beats_pop_ties'].sum())}/{n},"
              f" by more than its interval at "
              f"{int((t['lift_lo_ties'].astype(float) > 1).sum())}/{n}")
    if has_path:
        # NOTES §34.1 judges these at the seven reference dates
        # (scripts/path_verdict.py); here they are read, not ruled on.
        ratio = t["lift_vs_path"].astype(float)
        print(f"  beats path length alone (shortest import path first, ties "
              f"averaged) at {int(t['beats_path'].sum())}/{n} on PR-AUC;\n"
              f"  model/path median {ratio.median():.4f}x, range "
              f"{ratio.min():.4f}x – {ratio.max():.4f}x; and on nDCG@20, "
              f"within an upgrade,\n  at {int(t['beats_path_ndcg'].sum())}/"
              f"{n}")
    if has_ship:
        # The ranker that ships (NOTES §36.2), read as the model's lift is
        # read above: the count, the count clear of its interval, the
        # median and the range.
        pl = t["path_lift_vs_pop"].astype(float)
        clear = int((t["path_lift_lo"].astype(float) > 1).sum())
        print(f"  path length alone, the ranker that ships (NOTES §36.2): "
              f"beats popularity at {int(t['path_beats_pop'].sum())}/{n},\n"
              f"  by more than its interval at {clear}/{n}; lift vs pop "
              f"median {pl.median():.2f}x, range {pl.min():.2f}x – "
              f"{pl.max():.2f}x")
    if "reach" in t:
        # Item 15 (NOTES §36). Not printed here, as when the verdict was
        # made (§36.2); ship_verdict.py rules only on the sweep kept aside
        # from that day, and refuses a sweep stamped with the convention.
        print("\n  item 15 (NOTES §36): the five rankers' numbers are in the "
              "file; none are printed here.\n  The verdict is §36.2's, made "
              "on the sweep kept aside; scripts/ship_verdict.py does not\n  "
              "rule on this one.")

    if wins == n:
        print("\n  ** Holds at every cut date. The result is not one lucky")
        print("  ** date, and that is the claim worth making — not the "
              "single\n  ** best number in the column.")
    elif wins == 0:
        print("\n  ** Never beats popularity. Whatever the single-split run "
              "said,\n  ** this does not survive. Report the baseline as the "
              "finding.")
    else:
        print(f"\n  ** Beats popularity at only {wins} of {n} cut dates. That "
              "is a\n  ** coin-flip dressed as a result. Do not quote the "
              "best split;\n  ** quote this ratio.")

    v = t["pr_auc"].astype(float)
    if v.median() and (v.max() - v.min()) > 0.5 * v.median():
        print("\n  ** The spread exceeds half the median. Any comparison "
              "between\n  ** two models closer together than that spread is "
              "unreadable —\n  ** including the label-vs-label comparisons.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refit at several cut dates and report the spread.")
    ap.add_argument("--label", default="label_alias",
                    choices=["label", "label_scoped", "label_alias"])
    ap.add_argument("--all-labels", action="store_true",
                    help="run all three and compare them honestly")
    ap.add_argument("--objective", default="lambdarank",
                    choices=["lambdarank", "binary"],
                    help="binary fits a classifier at each cut instead of "
                         "the ranker, and writes its own _binary file. "
                         "F8 compares it with the line; F25 with the "
                         "ranker.")
    ap.add_argument("--stopping", default="cv", choices=["cv", "holdout"],
                    help="cv picks the tree count by folds inside train; "
                         "holdout is the old single-slice rule. Run both to "
                         "see whether the fix actually fixed anything.")
    ap.add_argument("--at", default=None,
                    help="comma-separated cut dates (YYYY-MM-DD) to use "
                         "instead of the quantiles, e.g. the dates a sweep "
                         "before a fix printed. Written to a separate _at "
                         "file, which train.py quotes only for path "
                         "length's row, and only at the seven reference "
                         "dates (NOTES §36.3).")
    ap.add_argument("--relevance", default=SHIPPED_RELEVANCE,
                    choices=RELEVANCE,
                    help="what each cut's model is taught: binary, or "
                         "graded by how many packages use a change (F2, "
                         "NOTES §25). Written to a _graded file.")
    ap.add_argument("--tuning", default=SHIPPED_TUNING, choices=TUNING,
                    help="fixed: the shipped setting, its tree count "
                         "clamped at 20. cv: setting and tree count chosen "
                         "on each cut's own folds (F7, NOTES §30). Written "
                         "to a _tuned file.")
    ap.add_argument("--contest", action="store_true",
                    help="also fit and score item 15's rankers for "
                         "label_alias (NOTES §36), as the record was made; "
                         "the verdict is in (§36.2) and ship_verdict.py "
                         "rules only on the sweep kept aside, so this is "
                         "for reading and for the tests, not for a verdict; "
                         "each cut then fits twice")
    args = ap.parse_args()
    points = None
    if args.at:
        points = [d.strip() for d in args.at.split(",") if d.strip()]
        bad = [d for d in points if pd.isna(
            pd.to_datetime(d, errors="coerce", format="%Y-%m-%d"))]
        if bad:
            sys.exit(f"--at wants dates like 2025-09-29; not a date: {bad}")

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    # This script ignores the `split` column and cuts all rows at its own
    # dates, so before the freeze its late cuts tested on the newest
    # releases, which are now the holdout. The cut quantiles are taken over
    # dev upgrades only, and a stale features.csv that still holds the
    # holdout stops here.
    assert_no_holdout(df, "stability.py")
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    labels = (["label", "label_scoped", "label_alias"] if args.all_labels
              else [args.label])
    for label in labels:
        problem = (relevance_problem(df, label, args.objective,
                                     args.relevance)
                   or tuning_problem(args.tuning, args.relevance,
                                     args.stopping))
        if problem:
            sys.exit(f"stability.py: {problem}")

    where = (f"{len(points)} cut dates given with --at" if points
             else f"{len(CUTS)} cut points")
    print(f"\n{len(df):,} rows   {where}   "
          f"{len(feats)} features   objective {args.objective}   "
          f"relevance {args.relevance}   stopping {args.stopping}   "
          f"tuning {args.tuning}")
    print("Each cut refits the model AND the baselines, so lift is computed")
    print("within a split before anything is summarised.")

    # Item 15 (NOTES §36) was about the model that ships, so only a sweep
    # run the way it is can take the extra fits, and only when asked for:
    # the verdict is in (§36.2).
    shipped = (args.objective == "lambdarank" and args.stopping == "cv"
               and args.relevance == SHIPPED_RELEVANCE
               and args.tuning == SHIPPED_TUNING)
    contest = args.contest and shipped and "label_alias" in labels
    if contest:
        print("For label_alias each cut also fits the two reachability "
              "features the same way,\nfor item 15 (NOTES §36).")
    elif args.contest:
        print("--contest is for a sweep run as the shipped model is "
              "(label_alias, lambdarank,\nstopping cv, binary, tuning cv); "
              "this one is not, so item 15's rankers are not fitted.")

    summary = {}
    for label in labels:
        t = run_label(df, label, feats, args.objective, args.stopping,
                      points, relevance=args.relevance, tuning=args.tuning,
                      contest=contest and label == "label_alias")
        report(t, label)
        out = DATA / stability_name(label, args.stopping, args.objective,
                                    args.relevance, at=bool(points),
                                    tuning=args.tuning)
        t.to_csv(out, index=False)
        print(f"\n  saved -> {out}")
        ok = t[~t["skipped"]]
        if not ok.empty:
            summary[label] = ok

    if len(summary) > 1:
        print("\n" + "=" * 74)
        print("  LABEL vs LABEL, ACROSS ALL CUT DATES")
        print("=" * 74)
        # COMPARE ON THE CUTS ALL LABELS SHARE, NOT ON EACH LABEL'S OWN.
        #
        # A cut is skipped per label — it needs enough positives and enough
        # rankable pairs, and a sparse label runs out first. So the labels
        # end up judged on different exams, and lift_MIN is a WORST CASE:
        # the label that was spared the hardest cut wins by not sitting it.
        #
        # Measured 16 Sep. `label` was skipped at q=0.85 for having 9
        # rankable pairs, so it never faced 2026-08-22 — which is exactly
        # where `label_alias` recorded its worst case, 1.45x. On its own
        # five cuts `label` showed 1.81x and won. On the five cuts all
        # three share, `label_alias` shows 1.92x and wins. Same data, same
        # code, opposite conclusion, and only one of them is a comparison.
        #
        # This is the THIRD time this rule has produced a misleading answer
        # (§11.2 was the first, §13.2 the same confusion in the audit). The
        # pattern each time: a number that looks like a measurement of the
        # models is partly a measurement of which rows each one was given.
        common = None
        for t in summary.values():
            cuts = set(t["cut"])
            common = cuts if common is None else (common & cuts)
        common = common or set()
        if any(len(t) != len(common) for t in summary.values()):
            print(f"\n  Comparing on the {len(common)} cut date(s) all labels")
            print("  share. Per-label tables above use every cut that label")
            print("  could use; a worst case taken over different cuts is not")
            print("  a comparison — see the note in the source.")

        rows = []
        for label, t_all in summary.items():
            t = t_all[t_all["cut"].isin(common)] if common else t_all
            if t.empty:
                continue
            v, l_ = t["pr_auc"].astype(float), t["lift_vs_pop"].astype(float)
            rows.append({
                "label": label,
                "splits": len(t),
                "pr_auc_median": round(v.median(), 4),
                "lift_median": round(l_.median(), 2),
                # THE DECIDING COLUMN. Medians hide failure modes: two
                # labels can sit 0.08x apart on the median while one of
                # them collapses to 0.37x at a cut the other handles at
                # 1.48x. A model that is sometimes worse than the dumb
                # baseline is not "slightly behind on average", it is
                # unshippable — you cannot tell in advance which day you
                # are having.
                "lift_MIN": round(l_.min(), 2),
                "lift_max": round(l_.max(), 2),
                "beats_pop": f"{int(t['beats_pop'].sum())}/{len(t)}",
            })
        comp = pd.DataFrame(rows)
        print()
        print(comp.to_string(index=False))

        print("\nRead the LIFT column, not PR-AUC. PR-AUC's floor is the")
        print("positive rate and the three labels have different ones, so")
        print("the raw scores are not comparable even here.")

        by_median = comp.loc[comp["lift_median"].idxmax(), "label"]
        by_worst = comp.loc[comp["lift_MIN"].idxmax(), "label"]
        gaps = sorted(r["lift_median"] for r in rows)
        close = len(gaps) > 1 and (gaps[-1] - gaps[-2]) < 0.5

        if close:
            print("\n** Median lift does NOT separate these: the top two are")
            print("** within 0.5x of each other. Ignore that column.")
        print(f"\n** Decide on lift_MIN. '{by_worst}' has the best worst case "
              f"at\n** {comp.loc[comp['lift_MIN'].idxmax(), 'lift_MIN']:.2f}x, "
              f"and wins popularity at "
              f"{comp.loc[comp['lift_MIN'].idxmax(), 'beats_pop']} cut dates.")
        survivors = comp[comp["lift_MIN"] > 1.0]["label"].tolist()
        if survivors:
            print(f"** Never worse than the baseline at any cut: "
                  f"{', '.join(survivors)}.")
        losers = comp[comp["lift_MIN"] <= 1.0]["label"].tolist()
        if losers:
            print(f"** Sometimes LOSES to popularity: {', '.join(losers)}. "
                  "A model\n** that is worse than the dumb baseline on some "
                  "dates cannot be\n** shipped on the strength of its median.")
        if by_median != by_worst:
            print(f"\n** Note: '{by_median}' has the better median but "
                  f"'{by_worst}' has\n** the better worst case. Prefer the "
                  "worst case.")


if __name__ == "__main__":
    main()
