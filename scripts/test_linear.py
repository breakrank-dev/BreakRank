"""
Item 7, first half (F8): is the linear baseline what it says it is, and
is everything the ranker reports exactly what it was?

    python scripts/test_linear.py

No network and no real data; under a minute. It uses test_holdout.py's
labelled fixture and two small frames built here. Nothing in your data/
or artifacts/ is read or touched.

WHY THIS FILE EXISTS. A baseline that is quietly wrong is worse than no
baseline: a line fitted on test rows, a line that ties every row of a
release, or a classifier scored by its 0/1 labels would each print a
number that reads as a comparison and is not one.

Eight cases:

  1. The line is fitted on TRAIN ONLY: flipping test labels moves no
     score. It is one fit, so two fits agree to the last digit, and it
     varies inside a release, which semver and popularity cannot.
  2. A straight signal is learned: positives that follow a logistic
     function of three features put the line well above the floor.
  3. A middle bump is not: positives that sit at one middle value of
     prior_breaks_in_module, with both ends quiet, leave the line at the
     floor while twenty trees find them. F8's mechanism, on data built
     to have it; whether the real data has it is what item 7 measures.
  4. Values the training rows never showed: a kind unseen in train, a
     missing number, booleans as bool, int or the text "True"/"False" all
     score, and score the same.
  5. Nothing the ranker reports changes: the four earlier baselines come
     out of add_baseline_scores exactly as before, the ranker's scores
     are its predict() as before, and a classifier's scores are now its
     probabilities, never its labels (train.score_with).
  6. The rule in scripts/item7_linear.py, at every edge: 6 of 7 wins and
     5 does not; a median of exactly 1.25x counts and 1.24x does not; the
     mirror for the line at 0.80x; a tie is neither a win nor a loss;
     mismatched columns are refused.
  7. The sweep's row carries the line, with its ratio and flag agreeing
     with its PR-AUC, and the lambdarank sweep file keeps the name every
     earlier file had while a binary sweep gets its own.
  8. baselines.py end to end on the fixture, in a temp directory: the
     line is in its table and named in its summary.
"""

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.build import (BOOLEAN, CATEGORICAL, NUMERIC,  # noqa: E402
                               add_features, drop_version_strings)
from ml.holdout import GROUP, drop_holdout  # noqa: E402
from ml.model import baselines as B  # noqa: E402
from ml.model import stability  # noqa: E402
from ml.model import train as T  # noqa: E402
from ml.model.metrics import evaluate  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []
FEATS = NUMERIC + BOOLEAN + CATEGORICAL
LABEL = "label_alias"


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- fixtures

def fixture_halves():
    """test_holdout.py's dev rows, features built, cut at 15 May like
    test_intervals.py: a realistic train and test half."""
    dev = T.prepare(drop_holdout(add_features(drop_version_strings(
        load("test_holdout").make_labelled())))).sort_values(GROUP)
    when = pd.to_datetime(dev["released_at"])
    return dev[when <= "2026-05-15"], dev[when > "2026-05-15"]


def synthetic(seed: int, shape: str) -> pd.DataFrame:
    """40 upgrades of 100 changes, every feature noise, and a label that is
    either a logistic function of three features ("straight") or 60% at
    prior_breaks_in_module == 3 and 1% at 0 and at 30 ("bump")."""
    rng = np.random.default_rng(seed)
    n_up, per = 40, 100
    n = n_up * per
    df = pd.DataFrame({
        "package": np.repeat([f"p{i}" for i in range(n_up)], per),
        "version_from": "1.0", "version_to": "1.1",
        "symbol": [f"s{i}" for i in range(n)],
        "module_depth": rng.integers(1, 5, n),
        "public_depth": rng.integers(1, 4, n),
        "inherited_by": rng.choice([0, 0, 0, 0, 2, 10], n),
        "prior_breaks_in_module": rng.choice([0, 3, 30], n,
                                             p=[0.4, 0.2, 0.4]),
        "name_length": rng.integers(3, 30, n),
        "package_rank": np.repeat(rng.integers(1, 500, n_up), per),
        "release_size": per,
        "package_churn": rng.integers(0, 50, n),
        "kind": rng.choice(["removed", "changed", "deprecated"], n),
        "bump": rng.choice(["major", "minor", "patch"], n)})
    for b in BOOLEAN:
        df[b] = rng.integers(0, 2, n)
    if shape == "straight":
        z = (-3 + 1.2 * (df["public_depth"] - 2) - 1.5 * df["is_private"]
             + 0.8 * df["in_dunder_all"])
        p = 1 / (1 + np.exp(-z))
    else:
        p = np.where(df["prior_breaks_in_module"] == 3, 0.6, 0.01)
    df[LABEL] = (rng.random(n) < p).astype(int)
    df["split"] = np.where(np.arange(n) < n // 2, "train", "test")
    return T.prepare(df)


def halves(df: pd.DataFrame):
    return df[df.split == "train"], df[df.split == "test"]


# -------------------------------------------------------------------- cases

def case_train_only(train, test) -> None:
    print("\n1. FITTED ON TRAIN ONLY, ONCE, AND IT RANKS INSIDE A RELEASE")
    a = B.linear_scores(train, test, LABEL)
    flipped = test.assign(**{LABEL: 1 - test[LABEL].astype(int)})
    b = B.linear_scores(train, flipped, LABEL)
    check("flipping every test label moves no score",
          bool(np.array_equal(a, b)), True)
    c = B.linear_scores(train, test, LABEL)
    check("two fits agree to the last digit", bool(np.array_equal(a, c)),
          True)
    check("every score is a probability", bool((a >= 0).all()
                                              and (a <= 1).all()), True)
    sc = test.assign(linear=a)
    within = int(sc.groupby(GROUP)["linear"].nunique().max())
    check("scores differ inside a release (not pinned like semver)",
          within > 1, True)
    try:
        B.linear_scores(train[train[LABEL] == 0], test, LABEL)
        refused = False
    except ValueError:
        refused = True
    check("a train half with one class is refused, not scored", refused,
          True)


def case_straight() -> None:
    print("\n2. A STRAIGHT SIGNAL IS LEARNED")
    train, test = halves(synthetic(0, "straight"))
    sc = test.assign(linear=B.linear_scores(train, test, LABEL))
    floor = float(test[LABEL].mean())
    pr = evaluate(sc, "linear", LABEL)["pr_auc"]
    check(f"the line is above 3x the floor ({pr:.3f} vs {floor:.3f})",
          pr > 3 * floor, True)


def case_bump() -> None:
    print("\n3. A MIDDLE BUMP IS NOT, AND TREES FIND IT")
    train, test = halves(synthetic(0, "bump"))
    sc = test.assign(linear=B.linear_scores(train, test, LABEL))
    clf = T.fit_fixed(train, FEATS, LABEL, 20, objective="binary")
    sc["trees"] = T.score_with(clf, test, FEATS)
    floor = float(test[LABEL].mean())
    line = evaluate(sc, "linear", LABEL)["pr_auc"]
    trees = evaluate(sc, "trees", LABEL)["pr_auc"]
    check(f"the line stays under 1.5x the floor ({line:.3f} vs "
          f"{floor:.3f})", line < 1.5 * floor, True)
    check(f"twenty trees clear 3x the floor ({trees:.3f})",
          trees > 3 * floor, True)


def case_unseen(train, test) -> None:
    print("\n4. VALUES TRAINING NEVER SHOWED")
    base = B.linear_scores(train, test, LABEL)
    odd = test.copy()
    odd["kind"] = odd["kind"].astype(str)
    odd.iloc[0, odd.columns.get_loc("kind")] = "never_seen_kind"
    odd.iloc[1, odd.columns.get_loc("package_rank")] = np.nan
    s = B.linear_scores(train, odd, LABEL)
    check("an unseen kind and a missing number both score, finitely",
          bool(np.isfinite(s[:2]).all()), True)
    check("and no other row moved", bool(np.allclose(s[2:], base[2:])), True)

    as_bool = train.copy()
    for b in BOOLEAN:
        as_bool[b] = as_bool[b].astype(bool)
    as_text = train.copy()
    for b in BOOLEAN:
        as_text[b] = as_text[b].astype(bool).astype(str)
    s_bool = B.linear_scores(as_bool, test, LABEL)
    s_text = B.linear_scores(as_text, test, LABEL)
    check("booleans as bool, as int and as the text True/False fit the "
          "same line",
          bool(np.allclose(s_bool, base) and np.allclose(s_text, base)),
          True)


def case_unchanged(train, test) -> None:
    print("\n5. NOTHING THE RANKER REPORTS CHANGES")
    sc = B.add_baseline_scores(train, test, LABEL)
    check("the five baseline columns are there",
          [c for c in ("griffe_all", "semver", "popularity", "kind_prior",
                       "linear") if c in sc], ["griffe_all", "semver",
                                               "popularity", "kind_prior",
                                               "linear"])
    semver = test["bump"].astype(str).map(B.BUMP_SCORE).fillna(0.0)
    pop = -test["package_rank"].fillna(test["package_rank"].max())
    rates = train.groupby("kind", observed=True)[LABEL].mean()
    rates.index = rates.index.astype(str)
    prior = (test["kind"].astype(str).map(rates)
             .fillna(train[LABEL].mean()))
    check("griffe_all, semver, popularity and kind_prior are the formulas "
          "they were",
          bool((sc["griffe_all"] == 1.0).all()
               and np.allclose(sc["semver"], semver)
               and np.allclose(sc["popularity"], pop)
               and np.allclose(sc["kind_prior"], prior)), True)

    ranker = T.fit_fixed(train, FEATS, LABEL, 20)
    check("the ranker's scores are its predict(), as before",
          bool(np.array_equal(T.score_with(ranker, test, FEATS),
                              ranker.predict(test[FEATS]))), True)
    clf = T.fit_fixed(train, FEATS, LABEL, 20, objective="binary")
    s = T.score_with(clf, test, FEATS)
    check("a classifier's scores are its probabilities",
          bool(np.array_equal(s, clf.predict_proba(test[FEATS])[:, 1])),
          True)
    check("and not its 0/1 labels", bool(len(set(np.round(s, 6))) > 2),
          True)


def case_rule() -> None:
    print("\n6. THE RULE, AT EVERY EDGE")
    I = load("item7_linear")
    line = [0.20] * 7

    def out(ratios):
        return I.verdict([0.20 * r for r in ratios], line)["outcome"]

    check("7 of 7 at 1.5x: trees", out([1.5] * 7), "trees")
    check("6 of 7 at 1.3x, one date lost: trees",
          out([1.3] * 6 + [0.9]), "trees")
    check("5 of 7 at 1.5x, two dates lost: no difference",
          out([1.5] * 5 + [0.9, 0.9]), "none")
    check("7 of 7 but the median is 1.10x: no difference",
          out([1.1, 1.1, 1.1, 1.1, 1.6, 1.6, 1.6]), "none")
    check("a median of exactly 1.25x counts",
          out([1.25] * 7), "trees")
    check("and 1.24x does not", out([1.24] * 7), "none")
    check("the line, 6 of 7 at 0.7x: line", out([0.7] * 6 + [1.1]), "line")
    check("a median of exactly 0.80x counts for the line",
          out([0.8] * 7), "line")
    check("and 0.81x does not", out([0.81] * 7), "none")
    check("a tie is neither a win nor a loss",
          I.verdict([0.2] * 7, line)["wins"]
          + I.verdict([0.2] * 7, line)["losses"], 0)
    check("the median is reported with the outcome",
          round(I.verdict([0.3, 0.2, 0.4, 0.2, 0.2, 0.5, 0.2], line)
                ["median"], 2), 1.0)
    try:
        I.verdict([0.2] * 6, line)
        refused = False
    except ValueError:
        refused = True
    check("six values against seven is refused", refused, True)
    check("the dates and shipped lifts are §23.5's and §24.2's",
          (len(I.DATES), I.SHIPPED[0], I.SHIPPED[-1], I.NEED, I.MARGIN),
          (7, 4.42, 2.30, 6, 1.25))


def case_sweep_row(dev: pd.DataFrame) -> None:
    print("\n7. THE SWEEP'S ROW, AND THE FILE NAMES")
    row = stability.one_split(dev, 0.70, LABEL, FEATS, "lambdarank")
    has = row is not None and not row.get("skipped", True)
    check("one cut of the fixture fits", has, True)
    if has:
        check("the row carries the line's PR-AUC and nDCG@20, its ratio "
              "and its flag",
              [k in row for k in ("linear", "linear_ndcg_20",
                                  "lift_vs_linear", "beats_linear")],
              [True] * 4)
        check("the ratio is the model's PR-AUC over the line's, to 2 "
              "places",
              row["lift_vs_linear"],
              round(row["pr_auc"] / row["linear"], 2))
        check("the flag is the comparison",
              row["beats_linear"], row["pr_auc"] > row["linear"])
    close = {"pr_auc": 0.2001, "ndcg_at_20": 0.5}
    same = {"pr_auc": 0.2000, "ndcg_at_20": 0.4}
    check("a hair ahead is ahead, a tie is not, and the line's numbers are "
          "the line's",
          (stability.against_the_line(close, same)["beats_linear"],
           stability.against_the_line(same, same)["beats_linear"],
           stability.against_the_line(close, same)["linear"],
           stability.against_the_line(close, same)["linear_ndcg_20"]),
          (True, False, 0.2, 0.4))
    check("a line at zero gives no ratio rather than an error",
          np.isnan(stability.against_the_line(
              close, {"pr_auc": 0.0, "ndcg_at_20": 0.0})["lift_vs_linear"]),
          True)
    check("the lambdarank sweep keeps the name every earlier file had",
          T.stability_name("label_alias", "cv"),
          "stability_label_alias_cv.csv")
    check("a binary sweep gets its own",
          T.stability_name("label_alias", "cv", "binary"),
          "stability_label_alias_cv_binary.csv")
    check("objective, then graded, then at",
          T.stability_name("label", "holdout", "binary", "graded", True),
          "stability_label_holdout_binary_graded_at.csv")
    check("and a graded lambdarank file is what it was",
          T.stability_name("label_alias", "cv", "lambdarank", "graded"),
          "stability_label_alias_cv_graded.csv")


def case_script(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n8. baselines.py END TO END")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-linear-"))
    try:
        (tmp / "data").mkdir()
        dev = pd.concat([train.assign(split="train"),
                         test.assign(split="test")]).sort_values(GROUP)
        dev.to_csv(tmp / "data" / "features.csv", index=False)
        p = subprocess.run([sys.executable,
                            str(ROOT / "ml/model/baselines.py")],
                           cwd=tmp, capture_output=True, text=True,
                           timeout=600)
        out = p.stdout + p.stderr
        check("it runs", p.returncode, 0)
        check("the line is in the table", "linear" in out, True)
        check("and named in the summary with the feature count",
              f"the line (logistic regression, the ranker's {len(FEATS)} "
              "features)" in out, True)
        saved = tmp / "data" / f"baselines_{LABEL}.csv"
        check("the saved table has a linear row",
              saved.exists() and "linear" in pd.read_csv(saved, index_col=0)
              .index, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    train, test = fixture_halves()
    case_train_only(train, test)
    case_straight()
    case_bump()
    case_unseen(train, test)
    case_unchanged(train, test)
    case_rule()
    case_sweep_row(pd.concat([train, test]).sort_values(GROUP))
    case_script(train, test)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The line is fitted on train only, finds a")
    print("straight signal and not a bump, and nothing the ranker reports")
    print("has changed.")


if __name__ == "__main__":
    main()
