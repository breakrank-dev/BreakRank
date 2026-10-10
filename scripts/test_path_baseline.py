"""
F41 and F42 (NOTES §34, §35): is path length alone scored the way §34.1
says, is average precision with ties averaged exact, does every sweep row
and train.py carry both, and does scripts/path_verdict.py apply §34.1's
rule as written?

    python scripts/test_path_baseline.py

No network and no real data; about two minutes. Small hand-made scores,
the labelled fixture test_holdout.py builds, a real build.py, train.py
and final_eval.py on it in a temp directory, and hand-made sweep files.

Six cases:

  1. Average precision with ties averaged (F42): the mean over every
     order of the ties, checked against all of them on small cases; the
     ordinary average precision when nothing ties; a row's weight the same
     as that many copies of it; a constant score at its closed form; the
     example the review found, 0.667 where sklearn reads 0.5; and one
     resample of the intervals, recomputed by hand with ties averaged.
  2. The baseline: minus public_depth, nothing fitted (flipping every
     training label moves no score), a row with no depth scored as the
     deepest, and its PR-AUC taken with ties averaged.
  3. tie_averaged(), where every tie-averaged number comes from, against
     the same numbers worked out here on a test half whose model scores
     tie; then the sweep's row: the F42 columns, path's three numbers, the
     ratio of the two four-place values to four places, and flags compared
     at four places, so a tie there is ahead for neither; the label and
     objective stamped on every row; and the sweep's report reads them.
  4. train.py and final_eval.py, on a fixture whose model scores tie: the
     tie-averaged block, path with the paired interval of the model's lift
     over it, and all of it in model_run's notes and the NOTES block.
  5. The rule, at every edge, on both measures, with ratios read to four
     places.
  6. The verdict on files: both verdicts printed from a good sweep, and a
     refusal for a sweep without the path columns, of another label or
     objective, at other dates, with a skipped date, or run another way.
     Nothing is written.
"""

import contextlib
import hashlib
import importlib.util
import io
import itertools
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.build import (BOOLEAN, CATEGORICAL,  # noqa: E402
                               NUMERIC, add_features, drop_version_strings)
from ml.holdout import GROUP, HOLDOUT_START, drop_holdout  # noqa: E402
from ml.model import stability  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import (PAIR, average_precision_ties,  # noqa: E402
                              evaluate, intervals, tie_averaged)
from ml.model.train import fit_fixed, prepare, score_with  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []
LABEL = "label_alias"
FEATS = NUMERIC + BOOLEAN + CATEGORICAL
DATES = ["2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
         "2026-03-02", "2026-04-02", "2026-05-04"]


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


def every_order(y, s) -> float:
    """Average precision averaged over every order of the tied scores, by
    listing them all: the definition, for small cases only."""
    y, s = np.asarray(y), np.asarray(s, float)
    runs = {}
    for i, v in enumerate(s):
        runs.setdefault(v, []).append(i)
    blocks = [list(itertools.permutations(runs[v]))
              for v in sorted(runs, reverse=True)]
    total, n = 0.0, 0
    for pick in itertools.product(*blocks):
        order = [i for b in pick for i in b]
        rank = np.empty(len(y))
        rank[order] = np.arange(len(y), 0, -1)
        total += average_precision_score(y, rank)
        n += 1
    return total / n


def case_ties() -> None:
    print("\n1. AVERAGE PRECISION WITH TIES AVERAGED (F42)")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(2, 8))
        y = rng.integers(0, 2, n)
        y[0] = 1
        s = rng.integers(0, 3, n).astype(float)
        worst = max(worst, abs(average_precision_ties(y, s)
                               - every_order(y, s)))
    check("the mean over every order of the ties, on 200 small cases",
          worst < 1e-12, True)
    y = rng.integers(0, 2, 3000)
    s = rng.random(3000)
    check("with nothing tied it is the ordinary average precision",
          abs(average_precision_ties(y, s) - average_precision_score(y, s))
          < 1e-12, True)
    worst = 0.0
    for _ in range(100):
        n = int(rng.integers(2, 5))
        y = rng.integers(0, 2, n)
        y[0] = 1
        s = rng.integers(0, 2, n).astype(float)
        w = rng.integers(1, 3, n)
        if w.sum() > 8:
            continue
        worst = max(worst, abs(average_precision_ties(y, s, sample_weight=w)
                               - every_order(np.repeat(y, w),
                                             np.repeat(s, w))))
    check("a row of weight k counts as k copies of itself", worst < 1e-12,
          True)
    big, pos = 2000, 150
    y = np.r_[np.ones(pos), np.zeros(big - pos)]
    h = (1 / np.arange(1, big + 1)).sum()
    check("a constant score: (1/N)(H_N + (R-1)/(N-1)(N - H_N))",
          round(average_precision_ties(y, np.zeros(big)), 10),
          round((h + (pos - 1) / (big - 1) * (big - h)) / big, 10))
    check("the review's example: 0.667, where sklearn reads 0.5",
          (round(average_precision_ties([1, 0, 1, 0], [-1, -1, -2, -2]), 4),
           round(average_precision_score([1, 0, 1, 0], [-1, -1, -2, -2]), 4)),
          (0.6667, 0.5))
    try:
        average_precision_ties([1, 0], [1, 0], sample_weight=[0.5, 1])
        refused = False
    except ValueError:
        refused = True
    check("weights that are not whole numbers are refused", refused, True)

    ups = np.repeat(np.arange(30), 5)
    df = pd.DataFrame({"package": [f"p{u}" for u in ups],
                       "version_from": "1.0", "version_to": "1.1",
                       LABEL: (rng.random(150) < 0.2).astype(int),
                       "model": rng.random(150),
                       "base": rng.integers(0, 3, 150).astype(float)})
    df.loc[0, LABEL] = 1
    key = df[PAIR].astype(str).agg("\x1f".join, axis=1)
    codes, uniq = pd.factorize(key)
    same, apart = [], []
    for seed in (1, 2, 3):
        got = intervals(df, LABEL, "model", "base", draws=1, seed=seed,
                        ties=True, per_upgrade=False)
        w = np.random.default_rng(seed).multinomial(
            len(uniq), np.full(len(uniq), 1 / len(uniq)), size=1)[0][codes]
        y = df[LABEL].to_numpy(int)
        want = average_precision_ties(y, df["base"], sample_weight=w)
        old = average_precision_score(y, df["base"], sample_weight=w)
        same.append(abs(got["baseline_pr_auc"][0] - want) < 1e-12)
        apart.append(abs(want - old) > 1e-6)
    check("one resample of intervals(ties=True), recomputed by hand, "
          "differs from sklearn's", (same, apart), ([True] * 3, [True] * 3))


def labelled() -> pd.DataFrame:
    return load("test_holdout").make_labelled()


def halves() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fixture's dev rows, split at 15 May, with a model's scores on
    the later half."""
    dev = prepare(drop_holdout(add_features(drop_version_strings(
        labelled())))).sort_values(GROUP)
    when = pd.to_datetime(dev["released_at"])
    train, test = dev[when <= "2026-05-15"], dev[when > "2026-05-15"]
    model = fit_fixed(train, FEATS, LABEL, 20)
    return train, test.assign(model=score_with(model, test, FEATS))


def case_baseline(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n2. THE BASELINE: SHORTEST IMPORT PATH FIRST, NOTHING FITTED")
    sc = add_baseline_scores(train, test, LABEL)
    depth = pd.to_numeric(test["public_depth"]).astype(float)
    check("path is minus public_depth", bool(np.allclose(sc["path"], -depth)),
          True)
    flipped = train.assign(**{LABEL: 1 - train[LABEL]})
    check("flipping every training label moves no path score",
          bool(np.array_equal(add_baseline_scores(flipped, test,
                                                  LABEL)["path"],
                              sc["path"])), True)
    gap = test.assign(public_depth=test["public_depth"].astype(float))
    gap.loc[gap.index[0], "public_depth"] = np.nan
    got = add_baseline_scores(train, gap, LABEL)["path"]
    check("a row with no depth scores as the deepest in its test half",
          float(got.iloc[0]), -float(depth.drop(gap.index[0]).max()))
    check("its PR-AUC with ties averaged is average_precision_ties on "
          "minus the depth",
          round(evaluate(sc, "path", LABEL, ties=True)["pr_auc"], 10),
          round(average_precision_ties(sc[LABEL], -depth), 10))
    check("and it can rank inside an upgrade: depth varies within one",
          bool(sc.groupby(GROUP)["path"].nunique().max() > 1), True)


def with_copies(df: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """The frame with n of its rows again, a parameter's worth apart: the
    same features and label, so the model scores each copy as it scores
    the row, and its scores tie, as a symbol's parameter rows do in the
    real data."""
    extra = df.sample(n, random_state=0).assign(sub_target="copy")
    return pd.concat([df, extra]).sort_values(GROUP, kind="mergesort")


def parameter_copies(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """The labelled frame with k more rows for every parameter row, each
    naming another parameter of the same symbol: every feature the same,
    so the model scores them alike and its scores tie in blocks, as a
    symbol's parameter rows do in the real data."""
    par = df[df["sub_target"].eq("arg")]
    return pd.concat([df] + [par.assign(sub_target=f"arg{i}")
                             for i in range(1, k + 1)], ignore_index=True)


def case_row(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n3. tie_averaged(), THE SWEEP'S ROW, ITS STAMPS AND ITS REPORT")
    sc = add_baseline_scores(train, with_copies(test), LABEL)
    y = sc[LABEL].to_numpy(int)
    tie = tie_averaged(sc, LABEL)
    ci = {(b, t): intervals(sc, LABEL, "model", b, ties=t,
                            per_upgrade=False)["lift"]
          for b in ("popularity", "path") for t in (True, False)}
    check("tie_averaged: the model's, popularity's and path's PR-AUC are "
          "average_precision_ties on their scores",
          [abs(tie[k] - average_precision_ties(y, sc[c])) < 1e-12
           for k, c in (("model", "model"), ("popularity", "popularity"))]
          + [abs(tie["path"]["pr_auc"] - average_precision_ties(y, sc["path"]))
             < 1e-12], [True] * 3)
    check("and the model's scores tie here, so its ordinary PR-AUC differs",
          abs(tie["model"] - average_precision_score(y, sc["model"])) > 1e-6,
          True)
    check("each lift is its two PR-AUCs' ratio",
          (abs(tie["lift_pop"] - tie["model"] / tie["popularity"]) < 1e-12,
           abs(tie["lift_path"] - tie["model"] / tie["path"]["pr_auc"])
           < 1e-12), (True, True))
    check("and each interval is intervals(ties=True)'s, not sklearn's",
          (tie["ci_pop"] == ci[("popularity", True)],
           tie["ci_path"] == ci[("path", True)],
           ci[("popularity", True)] != ci[("popularity", False)],
           ci[("path", True)] != ci[("path", False)]),
          (True, True, True, True))

    dev = pd.concat([train, test.drop(columns="model")])
    row = stability.one_split(dev, 0.70, LABEL, FEATS, "lambdarank")
    has = row is not None and not row.get("skipped", True)
    check("one cut of the fixture fits", has, True)
    keys = ("pr_auc_ties", "popularity_ties", "lift_vs_pop_ties",
            "lift_lo_ties", "lift_hi_ties", "beats_pop_ties", "path",
            "path_p_at_10", "path_ndcg_20", "lift_vs_path",
            "lift_vs_path_lo", "lift_vs_path_hi", "beats_path",
            "beats_path_ndcg")
    if has:
        check("the row carries the F42 columns and path's",
              [k for k in keys if k not in row], [])
        check("the F42 lift is the two tie-averaged PR-AUCs' ratio, inside "
              "its interval",
              (abs(row["lift_vs_pop_ties"] - row["pr_auc_ties"]
                   / row["popularity_ties"]) < 0.01,
               row["lift_lo_ties"] <= row["lift_vs_pop_ties"]
               <= row["lift_hi_ties"]), (True, True))
        check("the path ratio is the two stored values' ratio, to 4 places",
              row["lift_vs_path"],
              round(row["pr_auc_ties"] / row["path"], 4))
        check("the flags are the comparisons, as stored",
              (row["beats_path"], row["beats_path_ndcg"]),
              (row["pr_auc_ties"] > row["path"],
               row["ndcg_20"] > row["path_ndcg_20"]))
    m = {"pr_auc": 0.3, "ndcg_at_20": 0.50004, "precision_at_10": 0.3}
    p = {"pr_auc": 0.20001, "ndcg_at_20": 0.50001, "precision_at_10": 0.2}

    def given(model: float, path: dict) -> dict:
        return {"model": model, "path": path, "ci_path": (0.9, 1.3)}

    row_tie = stability.against_path(m, given(0.20004, p))
    check("a tie at four places is ahead for neither, on either measure",
          (row_tie["beats_path"], row_tie["beats_path_ndcg"]),
          (False, False))
    ahead = stability.against_path({**m, "ndcg_at_20": 0.5001},
                                   given(0.2001, p))
    check("one in the fourth place is ahead, on both",
          (ahead["beats_path"], ahead["beats_path_ndcg"]), (True, True))
    check("the PR-AUC side is the model's tie-averaged one, not its "
          "ordinary one (0.3 here), and the interval is the given one",
          (row_tie["lift_vs_path"], row_tie["lift_vs_path_lo"],
           row_tie["lift_vs_path_hi"]), (1.0, 0.9, 1.3))
    check("path's own numbers are path's",
          (row_tie["path"], row_tie["path_ndcg_20"],
           row_tie["path_p_at_10"]), (0.2, 0.5, 0.2))
    zero = stability.against_path(m, given(0.2, {**p, "pr_auc": 0.0}))
    check("a path PR-AUC of zero gives no ratio rather than an error",
          bool(np.isnan(zero["lift_vs_path"])), True)
    cols = stability.with_ties({"model": 0.3, "popularity": 0.12,
                                "lift_pop": 2.5, "ci_pop": (1.9, 3.3)})
    check("the F42 columns are tie_averaged()'s numbers, rounded",
          (cols["pr_auc_ties"], cols["popularity_ties"],
           cols["lift_vs_pop_ties"], cols["lift_lo_ties"],
           cols["lift_hi_ties"], cols["beats_pop_ties"]),
          (0.3, 0.12, 2.5, 1.9, 3.3, True))
    t = stability.run_label(dev, LABEL, FEATS, "lambdarank", "cv",
                            ["2020-01-01"])
    check("run_label stamps the label and the objective on every row",
          (list(t["label"]), list(t["objective"])),
          ([LABEL], ["lambdarank"]))

    t = pd.DataFrame({
        "cut": ["2026-01-01", "2026-02-01"], "q": [0.6, 0.7],
        "test_rows": 100, "test_pos": 40, "floor": 0.1, "trees": 20,
        "rankable10": 12, "pr_auc": 0.3, "popularity": 0.2,
        "lift_vs_pop": 1.5, "p_at_10": 0.3, "ndcg_20": 0.6,
        "beats_pop": True, "beats_semver": True, "skipped": False,
        "pr_auc_ties": 0.3, "popularity_ties": [0.24, 0.26],
        "lift_vs_pop_ties": [1.25, 1.15], "lift_lo_ties": [1.1, 0.9],
        "lift_hi_ties": 1.6, "beats_pop_ties": [True, False],
        "path": [0.25, 0.32], "path_p_at_10": 0.2,
        "path_ndcg_20": [0.55, 0.58], "lift_vs_path": [1.2, 0.9375],
        "beats_path": [True, False], "beats_path_ndcg": [True, True]})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stability.report(t, LABEL)
    out = buf.getvalue()
    check("the report reads them: the F42 lift, and path at 1/2 on PR-AUC "
          "with a four-place median, 2/2 on nDCG@20",
          ("with ties averaged (F42): lift vs pop median 1.20x" in out,
           "beats popularity\n  at 1/2, by more than its interval at 1/2"
           in out,
           bool(re.search(r"ties averaged\) at 1/2 on PR-AUC;\n  model/path "
                          r"median 1\.0688x", out)),
           "within an upgrade,\n  at 2/2" in out,
           bool(re.search(r"^\s*cut\s.*\bpr_auc_ties\b.*\bpath\b.*"
                          r"\bpath_ndcg_20\b", out, re.M))),
          (True, True, True, True, True))


def case_scripts(tmp: pathlib.Path) -> None:
    print("\n4. train.py AND final_eval.py")
    (tmp / "data").mkdir()
    parameter_copies(labelled()).to_csv(tmp / "data" / "labelled.csv",
                                        index=False)
    p = subprocess.run([sys.executable, str(ROOT / "ml/features/build.py")],
                       cwd=tmp, capture_output=True, text=True, timeout=900)
    check("build.py exits cleanly", p.returncode, 0)
    if p.returncode:
        print(p.stdout[-1500:] + p.stderr[-1500:])
        return
    p = subprocess.run([sys.executable, str(ROOT / "ml/model/train.py"),
                        "--tuning", "fixed"], cwd=tmp, capture_output=True,
                       text=True, timeout=900)
    out = p.stdout + p.stderr
    check("train.py exits cleanly", p.returncode, 0)
    if p.returncode:
        print(out[-2500:])
        return
    head = re.search(r"with ties averaged \(F42\): model PR-AUC ([0-9.]+), "
                     r"popularity ([0-9.]+), path length alone ([0-9.]+)",
                     out)
    block = out.split("with ties averaged (F42)")[-1]
    lifts = {name: re.search(name + r"\s+([0-9.]+)x\s+([0-9.]+)x to "
                             r"([0-9.]+)x", block)
             for name in ("lift over popularity", "lift over path length")}
    check("it prints the tie-averaged block, each lift inside its own "
          "interval",
          (bool(head), all(m and float(m.group(2)) <= float(m.group(1))
                           <= float(m.group(3)) for m in lifts.values())),
          (True, True))
    table = re.search(r"^model\s+([0-9.]+)\s", out, re.M)
    check("the model's scores tie in this fixture, so the table's ordinary "
          "PR-AUC and the block's differ",
          bool(head and table and table.group(1) != head.group(1)), True)
    if head and all(lifts.values()):
        check("and each lift is the block's own ratio, to two places",
              [abs(float(head.group(1)) / float(head.group(i))
                   - float(lifts[n].group(1))) < 0.006
               for i, n in ((2, "lift over popularity"),
                            (3, "lift over path length"))], [True, True])
    notes = pd.read_json(tmp / "artifacts" / "metrics.json",
                         typ="series")["notes"]
    found = re.search(r"pr_auc_ties=([0-9.]+) lift_vs_popularity_ties="
                      r"([0-9.]+)x lift_ties_95=([0-9.]+)-([0-9.]+) "
                      r"path=([0-9.]+) lift_vs_path=([0-9.]+)x "
                      r"lift_vs_path_95=([0-9.]+)-([0-9.]+)", notes)
    check("model_run's notes carry them all", bool(found), True)
    if found and head and all(lifts.values()):
        pl = lifts["lift over path length"]
        pp = lifts["lift over popularity"]
        check("the same numbers as printed",
              (found.group(1), found.group(2), found.group(3),
               found.group(4), found.group(5), found.group(6),
               found.group(7), found.group(8)),
              (head.group(1), pp.group(1), pp.group(2), pp.group(3),
               head.group(3), pl.group(1), pl.group(2), pl.group(3)))

    p = subprocess.run([sys.executable, str(ROOT / "ml/model/final_eval.py"),
                        "--unseal", "--tuning", "fixed"], cwd=tmp,
                       capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    hold = pd.read_csv(tmp / "data" / "holdout.csv")
    said = re.search(r"path length alone ([0-9.]+),", out)
    check("final_eval.py's path PR-AUC is path's on the holdout, ties "
          "averaged",
          said and float(said.group(1)),
          round(average_precision_ties(hold[LABEL],
                                       -hold["public_depth"]), 4))
    check("final_eval.py puts the tie-averaged numbers and path length "
          "alone in its NOTES block",
          (p.returncode, bool(re.search(
              r"With ties averaged \(F42\): PR-AUC [0-9.]+ vs popularity "
              r"[0-9.]+, [0-9.]+x \([0-9.]+-[0-9.]+\);\n  path length alone "
              r"[0-9.]+, [0-9.]+x \([0-9.]+-[0-9.]+\), its nDCG@20 [0-9.]+ "
              r"against the model's [0-9.]+\.", out))), (0, True))


def case_rule() -> None:
    print("\n5. THE RULE, AT EVERY EDGE")
    V = load("path_verdict")
    base = [0.20] * 7

    def acr(ratios):
        return V.across([0.20 * r for r in ratios], base)["outcome"]

    check("PR-AUC, 7 of 7 at 1.5x: the model", acr([1.5] * 7), "model")
    check("6 of 7 at 1.3x, one date lost: the model",
          acr([1.3] * 6 + [0.9]), "model")
    check("5 of 7 at 1.5x, two lost: no difference",
          acr([1.5] * 5 + [0.9, 0.9]), "none")
    check("7 of 7 but a median of 1.10x: no difference",
          acr([1.1, 1.1, 1.1, 1.1, 1.6, 1.6, 1.6]), "none")
    check("a median of exactly 1.25x counts", acr([1.25] * 7), "model")
    check("and 1.24x does not", acr([1.24] * 7), "none")
    check("path, 6 of 7 at 0.7x: path", acr([0.7] * 6 + [1.1]), "path")
    check("a median of exactly 0.80x counts for path", acr([0.8] * 7),
          "path")
    check("and 0.81x does not", acr([0.81] * 7), "none")
    check("ratios are read to four places, as the sweep stores and prints "
          "them: 0.6251 / 0.5001 reads 1.2500",
          V.across([0.6251] * 7, [0.5001] * 7)["outcome"], "model")
    check("and 0.3201 / 0.4001 reads 0.8000",
          V.across([0.3201] * 7, [0.4001] * 7)["outcome"], "path")
    tie = V.across(base, base)
    check("a tie is neither a win nor a loss", tie["wins"] + tie["losses"],
          0)
    nd = [0.55] * 7

    def wit(model):
        return V.within(model, nd)["outcome"]

    check("nDCG@20 higher at 6 of 7: the model",
          wit([0.56] * 6 + [0.54]), "model")
    check("higher at 5 of 7, a tie at the sixth: no difference",
          wit([0.56] * 5 + [0.55, 0.54]), "none")
    check("path higher at 6 of 7: path", wit([0.54] * 6 + [0.56]), "path")
    check("ties at all 7: no difference, and neither counts",
          (wit(nd), V.within(nd, nd)["ahead"] + V.within(nd, nd)["behind"]),
          ("none", 0))
    refused = []
    for f in (V.across, V.within):
        try:
            f([0.2] * 6, base)
            refused.append(False)
        except ValueError:
            refused.append(True)
    check("six values against seven is refused, on both", refused,
          [True, True])
    check("the dates are §23.5's, the bar 6 of 7 and 1.25x, the model the "
          "shipped one",
          (V.DATES, V.NEED, V.MARGIN, V.LABEL, V.OBJECTIVE),
          (DATES, 6, 1.25, LABEL, "lambdarank"))


def sweep(pr: list[float], path: list[float], nd: list[float],
          path_nd: list[float], dates: list[str] = DATES,
          **stamps) -> pd.DataFrame:
    n = len(pr)
    t = pd.DataFrame({
        "cut": dates[:n], "floor": 0.05,
        "pr_auc": [round(x - 0.01, 4) for x in pr], "pr_auc_ties": pr,
        "path": path, "ndcg_20": nd, "path_ndcg_20": path_nd,
        "p_at_10": 0.25, "path_p_at_10": 0.2, "lift_vs_pop": 3.0,
        "skipped": False, "holdout_from": str(HOLDOUT_START.date()),
        "relevance": "binary", "tuning": "cv", "label": LABEL,
        "objective": "lambdarank"})
    return t.assign(**stamps)


def verdict_on(tmp: pathlib.Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(ROOT / "scripts/path_verdict.py"),
                        "sweep.csv"], cwd=tmp, capture_output=True,
                       text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def run_verdict(tmp: pathlib.Path, t: pd.DataFrame) -> tuple[int, str]:
    t.to_csv(tmp / "sweep.csv", index=False)
    return verdict_on(tmp)


def case_files(tmp: pathlib.Path) -> None:
    print("\n6. THE VERDICT, ON FILES")
    tmp = tmp / "verdict"
    tmp.mkdir()
    pr = [0.30, 0.28, 0.26, 0.31, 0.22, 0.27, 0.29]
    good = sweep(pr, [round(x / 1.4, 4) for x in pr], [0.60] * 7,
                 [0.58] * 5 + [0.61, 0.60])
    good.to_csv(tmp / "sweep.csv", index=False)
    written = hashlib.sha256((tmp / "sweep.csv").read_bytes()).hexdigest()
    code, out = verdict_on(tmp)
    check("a good sweep gets both verdicts: the model across releases, and "
          "no difference within an upgrade (ahead at 5, a tie, behind at 1)",
          (code, "VERDICT: the model beats path length across releases"
           in out, "VERDICT: no difference the rule can see within an "
           "upgrade" in out, "the model higher at 5/7, path higher at 1/7"
           in out), (0, True, True, True))
    check("the PR-AUC it reads is the tie-averaged one, not pr_auc",
          "0.3000  0.2143" in out, True)
    check("nothing is written: the folder holds only the sweep, unchanged",
          {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in tmp.iterdir()}, {"sweep.csv": written})
    near = sweep([0.2997] + pr[1:],
                 [0.3001] + [round(x / 1.4, 4) for x in pr[1:]],
                 [0.60] * 7, [0.58] * 7)
    code, out = run_verdict(tmp, near)
    check("the table prints each ratio as the rule reads it, to four "
          "places: a date lost by 0.0004 reads 0.9987x, not 1.00x",
          (code, "0.9987x" in out, "1.00x" in out, "behind at 1/7" in out),
          (0, True, False, True))
    other = sweep(pr, pr, [0.6] * 7, [0.6] * 7)
    for name, t, says in (
            ("a sweep from before F41, with no path columns",
             good.drop(columns=["path", "path_ndcg_20", "path_p_at_10"]),
             "written before F41"),
            ("a sweep with no label or objective stamp",
             good.drop(columns=["label", "objective"]), "label is missing"),
            ("another label's sweep", other.assign(label="label"),
             "label is label, not label_alias"),
            ("a classifier's sweep", other.assign(objective="binary"),
             "objective is binary"),
            ("a sweep at other dates", sweep(pr, pr, [0.6] * 7, [0.6] * 7,
                                             DATES[:6] + ["2026-05-11"]),
             "not cut at the seven dates"),
            ("a sweep that skipped a date",
             good.assign(skipped=[False] * 6 + [True]), "skipped 2026-05-04"),
            ("a sweep that skipped every date",
             good[["cut", "skipped", "holdout_from", "relevance", "tuning",
                   "label", "objective"]].assign(skipped=True),
             "skipped 2025-08-07"),
            ("a sweep run with fixed trees", other.assign(tuning="fixed"),
             "tuning is fixed"),
            ("a sweep of graded models", other.assign(relevance="graded"),
             "relevance is graded"),
            ("a sweep with another holdout boundary",
             other.assign(holdout_from="2026-08-04"),
             "holdout_from is 2026-08-04")):
        code, out = run_verdict(tmp, t)
        check(f"refused: {name}", (code != 0, says in out, "VERDICT" in out),
              (True, True, False))
    (tmp / "sweep.csv").unlink()
    code, out = verdict_on(tmp)
    check("refused: no sweep at all, with the command that makes one",
          (code != 0, "stability.py --at " + ",".join(DATES) in out),
          (True, True))


def main() -> None:
    case_ties()
    train, test = halves()
    case_baseline(train, test)
    case_row(train, test)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-path-"))
    try:
        case_scripts(tmp)
        case_rule()
        case_files(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Ties are averaged exactly, and path length")
    print("alone is scored, reported and judged the way NOTES §34.1 says.")


if __name__ == "__main__":
    main()
