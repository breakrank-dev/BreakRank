"""
Item 15 (F25, NOTES §36): does the sweep fit and score the five rankers
the way §36.1 says, and does scripts/ship_verdict.py apply its rule as
written?

    python scripts/test_ship.py

No network and no real data; about four minutes. Small hand-made
scores, the labelled fixture test_holdout.py builds (its export flag
given signal of its own, so the two reachability features can rank
differently from path length), a real build.py and stability.py on it in
a temp directory, and hand-made sweep files.

Seven cases:

  1. then_by(): rows ordered by the first score, and only the rows it
     ties reordered by the second, however close the first score's values
     are; rows tied on both stay tied; a missing score refused.
  2. precision@10 and nDCG@20 averaged over every order of the ties
     (F44): against all of the orders on small cases, binary and graded;
     the fixed-order number where nothing ties; and the same whatever
     order the rows arrive in, which the fixed order is not. Then the
     test half's fingerprint (the same rows hash alike whatever pandas
     made of them) and the run's stamps, and F44's figures in NOTES §36
     reproduced on this fixture.
  3. The sweep's row, on the fixture: R is fitted inside the sweep on the
     two reachability features and nothing else, the way the model is,
     and ranks differently from path length; every number of the five
     rankers is the one worked out here; the model's numbers are the same
     with item 15 on or off; and all of it again under --stopping
     holdout.
  4. stability.py end to end: a sweep of the shipped model with --contest
     writes item 15's columns and prints none of their numbers; a sweep
     without it, or run another way, writes none; and the model's columns
     match a run without item 15, cell for cell. (The verdict is in,
     §36.2, so the contest's fits are made only when asked for.)
  5. The rule: the one test at its edges, and every branch of §36.1,
     both routes by which a draft would have shipped M without its
     beating P, or past a simpler ranker that beats it, included.
  6. The verdict on files: what ships from good sweeps; what it ran on;
     the model's numbers reported where they moved, not refused, with
     what that does to §34.1's list; and a refusal for a sweep from
     before item 15, with a gap or a missing column, not on §34.2's
     data, or run another way. Nothing is written.
  7. The constants: §23.5's dates, §34.1's functions, and §34.2's numbers,
     test rows and positives as NOTES records them.
"""

import hashlib
import importlib.util
import itertools
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.build import (BOOLEAN, CATEGORICAL,  # noqa: E402
                               NUMERIC, add_features, drop_version_strings)
from ml.holdout import HOLDOUT_START, drop_holdout  # noqa: E402
from ml.model import metrics, stability  # noqa: E402
from ml.model.ablate import REACHABILITY  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import evaluate  # noqa: E402
from ml.model.train import (FIXED_PARAMS, fit_cv,  # noqa: E402
                            fit_model, prepare, score_with, setting_text,
                            split_valid)

PASS, FAIL = "  ok  ", "  FAIL"
failures = []
LABEL = "label_alias"
FEATS = NUMERIC + BOOLEAN + CATEGORICAL
DATES = ["2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
         "2026-03-02", "2026-04-02", "2026-05-04"]
CUT = "2026-05-15"
CONTEST_KEYS = [c for cols in stability.CONTEST.values() for c in cols
                if c not in ("path", "pr_auc_ties")] + list(
                    stability.CONTEST_EXTRA)


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


def labelled() -> pd.DataFrame:
    """test_holdout.py's fixture, with an export flag that carries signal
    of its own. There it is `public_depth == 1` exactly, which makes the
    two reachability features rank as path length does and would let a
    broken R pass."""
    df = load("test_holdout").make_labelled()
    rng = np.random.default_rng(15)
    deep = df["public_depth"] > 1
    rate = np.where(df["label_alias"] == 1, 0.6, 0.1)
    df["has_export_path"] = ~deep | (rng.random(len(df)) < rate)
    return df


def norm(row: dict) -> dict:
    """A row with each NaN written "nan", so two rows compare equal when
    every value is the same, a missing one included."""
    return {k: ("nan" if isinstance(v, float) and np.isnan(v) else v)
            for k, v in row.items()}


# ------------------------------------------------------------- 1. then_by

def case_then_by() -> None:
    print("\n1. then_by(): THE FIRST SCORE, THEN THE SECOND INSIDE ITS TIES")
    s = stability.then_by([2, 2, 1, 1, 0], [0.1, 0.9, 0.5, 0.5, 0.3])
    check("by the first score, its ties by the second, rows tied on both "
          "still tied",
          (s[1] > s[0] > s[2], s[2] == s[3], s[3] > s[4]),
          (True, True, True))
    s = stability.then_by([-1, -1, -2], [-1e9, 1e9, 1e12])
    check("the second score never lifts a row past a better first score, "
          "however large", bool(s[1] > s[0] > s[2]), True)
    s = stability.then_by([0.30001, 0.30000, 0.30000], [0.0, 1.0, 0.0])
    check("first-score values 0.00001 apart still decide, as R's can",
          bool(s[0] > s[1] > s[2]), True)
    rng = np.random.default_rng(1)
    first = rng.integers(0, 4, 300).astype(float)
    second = rng.integers(0, 5, 300).astype(float)
    s = stability.then_by(first, second)
    f1, f2 = first[:, None], first[None, :]
    g1, g2 = second[:, None], second[None, :]
    above = (f1 > f2) | ((f1 == f2) & (g1 > g2))
    level = (f1 == f2) & (g1 == g2)
    check("on 300 random rows, every pair: one above another exactly when "
          "(first, second) is, level exactly when both are",
          (np.array_equal(s[:, None] > s[None, :], above),
           np.array_equal(s[:, None] == s[None, :], level)), (True, True))
    try:
        stability.then_by([1.0, np.nan], [0.1, 0.2])
        refused = False
    except ValueError:
        refused = True
    check("a missing score is refused", refused, True)


# ------------------------------------------------- 2. averaged tie orders

def every_order(score, rel, k: int, measure: str) -> float:
    """precision@k or nDCG@k averaged over every order of the tied scores,
    by listing them all: the definition, for small cases only."""
    score, rel = np.asarray(score, float), np.asarray(rel, float)
    runs = {}
    for i, v in enumerate(score):
        runs.setdefault(v, []).append(i)
    blocks = [list(itertools.permutations(runs[v]))
              for v in sorted(runs, reverse=True)]
    ideal = np.sort(rel)[::-1][:k]
    idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
    total, n = 0.0, 0
    for pick in itertools.product(*blocks):
        top = rel[[i for b in pick for i in b]][:k]
        if measure == "p":
            total += top.sum() / k
        else:
            total += float((top / np.log2(np.arange(2, len(top) + 2))).sum()
                           ) / idcg
        n += 1
    return total / n


def one_upgrade(score, rel, gain=None) -> pd.DataFrame:
    d = pd.DataFrame({"package": "p", "version_from": "1", "version_to": "2",
                      "s": score, "y": (np.asarray(rel) > 0).astype(int)})
    if gain is not None:
        d["g"] = gain
    return d


def case_expected() -> None:
    print("\n2. PRECISION@10 AND nDCG@20 AVERAGED OVER EVERY TIE ORDER (F44)")
    rng = np.random.default_rng(44)
    same_p, same_n, same_g = [], [], []
    for _ in range(40):
        n = int(rng.integers(5, 9))
        score = rng.integers(0, 3, n).astype(float)
        rel = (rng.random(n) < 0.4).astype(int)
        rel[0] = 1
        gain = rel * rng.choice([1, 3, 7, 15], n)
        k = 3
        d = one_upgrade(score, rel, gain)
        same_p.append(abs(metrics.precision_at_k(d, "s", "y", k,
                                                 expected=True)
                          - every_order(score, rel, k, "p")) < 1e-12)
        same_n.append(abs(metrics.ndcg_at_k(d, "s", "y", k, expected=True)
                          - every_order(score, rel, k, "n")) < 1e-12)
        same_g.append(abs(metrics.ndcg_at_k(d, "s", "y", k, gain="g",
                                            expected=True)
                          - every_order(score, gain, k, "n")) < 1e-12)
    check("on 40 small upgrades with ties: precision@k, nDCG@k and graded "
          "nDCG@k are the mean over every order", (all(same_p), all(same_n),
                                                    all(same_g)),
          (True, True, True))
    score = np.arange(30, dtype=float)[::-1]
    rel = (np.arange(30) % 4 == 0).astype(int)
    d = one_upgrade(score, rel)
    check("where nothing ties, the fixed-order numbers",
          (metrics.precision_at_k(d, "s", "y", 10, expected=True),
           metrics.ndcg_at_k(d, "s", "y", 20, expected=True)),
          (metrics.precision_at_k(d, "s", "y", 10, expected=False),
           metrics.ndcg_at_k(d, "s", "y", 20, expected=False)))
    score = np.repeat([3.0, 2.0, 1.0], 10)
    rel = (rng.random(30) < 0.3).astype(int)
    rel[:2] = 1
    d = one_upgrade(score, rel)
    shuffled = [d.sample(frac=1, random_state=i).reset_index(drop=True)
                for i in range(5)]
    averaged = {round(metrics.ndcg_at_k(x, "s", "y", 20, expected=True), 12)
                for x in shuffled + [d]}
    fixed = {round(metrics.ndcg_at_k(x, "s", "y", 20, expected=False), 12)
             for x in shuffled + [d]}
    check("and the same whatever order the rows come in, which the fixed "
          "order is not", (len(averaged), len(fixed) > 1), (1, True))
    rel = np.zeros(30, dtype=int)
    rel[[0, 8, 9, 15, 20, 29]] = 1
    d = one_upgrade(np.repeat([3.0, 2.0, 1.0], [7, 12, 11]), rel)
    e = evaluate(d, "s", "y", ties=True, expected=True)
    f = evaluate(d, "s", "y", ties=True, expected=False)
    check("evaluate(expected=True) carries both, here where a tie straddles "
          "the 10th and 20th places and the fixed order cannot match them",
          (e["precision_at_10"], e["ndcg_at_20"],
           e["precision_at_10"] != f["precision_at_10"],
           e["ndcg_at_20"] != f["ndcg_at_20"]),
          (metrics.precision_at_k(d, "s", "y", 10, expected=True),
           metrics.ndcg_at_k(d, "s", "y", 20, expected=True), True, True))
    check("and precision@10 there is the hand-worked 0.175: 7 places at "
          "1/7, then 3 at 3/12", round(e["precision_at_10"], 12), 0.175)


def case_spread(df: pd.DataFrame) -> None:
    """F44's numbers as NOTES §36 quotes them, from this fixture: path
    length's nDCG@20 at one cut under 30 fixed orders, its average over
    every order, and the order the code uses."""
    print("\n   F44's figures in NOTES §36, from this fixture")
    cutoff, is_test = stability.cut(df, "2026-04-01")
    test = df[is_test].sort_values(stability.GROUP)
    test = test.assign(path=-pd.to_numeric(test["public_depth"]).astype(float))
    seed, draws = metrics.TIE_SEED, []
    try:
        for s in range(30):
            metrics.TIE_SEED = s
            draws.append(metrics.ndcg_at_k(test, "path", LABEL, 20,
                                           expected=False))
    finally:
        metrics.TIE_SEED = seed
    check("89 upgrades rankable at 20; over 30 fixed orders 0.466 to 0.517; "
          "averaged over every order 0.488; the code's own order 0.496",
          (metrics.n_rankable(test, LABEL, 20), round(min(draws), 3),
           round(max(draws), 3),
           round(metrics.ndcg_at_k(test, "path", LABEL, 20, expected=True),
                 3), round(draws[0], 3)),
          (89, 0.466, 0.517, 0.488, 0.496))


def case_fingerprint() -> None:
    print("\n   THE TEST HALF'S FINGERPRINT AND THE RUN'S STAMPS")
    a = pd.DataFrame({"package": ["p", "p", "q"], "version_from": ["1", "1",
                                                                   "2"],
                      "version_to": ["2", "2", "3"],
                      "symbol": ["p.a", "p.b", "q.c"],
                      "sub_target": ["arg", np.nan, None],
                      LABEL: [1, 0, 0], "public_depth": [1, 2, 3],
                      "has_export_path": [1, 0, 1]})
    b = a.iloc[[2, 0, 1]].assign(
        public_depth=lambda x: x["public_depth"].astype(float),
        has_export_path=lambda x: x["has_export_path"].astype(bool),
        sub_target=["", "arg", pd.NA])
    c = a.assign(has_export_path=[1, 1, 1])
    check("the same rows hash alike whatever their order, number types or "
          "kind of missing value; one export flag changed does not",
          (stability.test_hash(a, LABEL) == stability.test_hash(b, LABEL),
           stability.test_hash(a, LABEL) == stability.test_hash(c, LABEL)),
          (True, False))
    env = stability.environment()
    try:
        status = subprocess.run(["git", "status", "--porcelain",
                                 "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        dirty = (bool(status.stdout.strip()) if status.returncode == 0
                 else None)
    except OSError:
        dirty = None
    check("the run's stamps: versions, CPUs, the OpenMP setting, and a "
          "commit that says when the tree had changes",
          (sorted(env), bool(re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?|unknown",
                                          env["commit"])),
           env["commit"] == "unknown" or dirty is None
           or env["commit"].endswith("-dirty") == dirty),
          (sorted(["lightgbm", "sklearn", "pandas", "numpy", "python",
                   "cpus", "omp_threads", "commit"]), True, True))


# ---------------------------------------------------- 3. the sweep's row

def dev() -> pd.DataFrame:
    return prepare(drop_holdout(add_features(drop_version_strings(
        labelled()))))


def worked_out(test: pd.DataFrame, model, reach, full_train: pd.DataFrame,
               trees: int, setting: str) -> tuple[dict, pd.DataFrame]:
    """Item 15's numbers for one test half, from fits made here."""
    sc = test.copy()
    sc["model"] = score_with(model, test, FEATS)
    sc = add_baseline_scores(full_train, sc, LABEL)
    sc["reach"] = score_with(reach, test, REACHABILITY)
    sc["path_model"] = stability.then_by(sc["path"], sc["model"])
    sc["reach_model"] = stability.then_by(sc["reach"], sc["model"])
    want = {"reach_trees": trees, "reach_setting": setting,
            "reach_levels": int(sc["reach"].nunique()),
            "test_hash": stability.test_hash(sc, LABEL)}
    for name in ("path", "reach", "model", "path_model", "reach_model"):
        e = evaluate(sc, name, LABEL, ties=True, expected=True)
        if name in ("reach", "path_model", "reach_model"):
            want[name] = round(e["pr_auc"], 4)
        prefix = "" if name == "model" else f"{name}_"
        want[f"{prefix}ndcg_20_e"] = round(e["ndcg_at_20"], 4)
        want[f"{prefix}p_at_10_e"] = round(e["precision_at_10"], 4)
    return want, sc


def case_row(df: pd.DataFrame) -> None:
    print("\n3. THE SWEEP'S ROW, ON THE FIXTURE")
    calls = []
    real = stability.fit_cv

    def spy(full_train, feats, label, objective="lambdarank",
            relevance="binary", tuning="fixed"):
        calls.append((list(feats), tuning, len(full_train)))
        return real(full_train, feats, label, objective,
                    relevance=relevance, tuning=tuning)

    stability.fit_cv = spy
    try:
        on = stability.one_split(df, CUT, LABEL, FEATS, "lambdarank", "cv",
                                 "binary", "cv", contest=True)
    finally:
        stability.fit_cv = real
    off = stability.one_split(df, CUT, LABEL, FEATS, "lambdarank", "cv",
                              "binary", "cv")
    check("inside the sweep R is fitted on the two reachability features "
          "alone, after the model, tuned and on the same rows as it",
          [(f, t) for f, t, _ in calls], [(FEATS, "cv"),
                                          (REACHABILITY, "cv")])
    check("on the same training rows", len({n for _, _, n in calls}), 1)
    check("item 15's columns only when asked for",
          ([k for k in CONTEST_KEYS if k not in on],
           [k for k in CONTEST_KEYS if k in off]), ([], []))
    check("and every other column the same with it on or off: the model's "
          "numbers do not depend on it",
          norm({k: v for k, v in on.items() if k not in CONTEST_KEYS}),
          norm(off))

    cutoff, is_test = stability.cut(df, CUT)
    test = df[is_test].sort_values(stability.GROUP)
    full_train = df[~is_test].sort_values(stability.GROUP)
    model, _, _ = fit_cv(full_train, FEATS, LABEL, "lambdarank",
                         relevance="binary", tuning="cv")
    reach, trees, _ = fit_cv(full_train, REACHABILITY, LABEL, "lambdarank",
                             relevance="binary", tuning="cv")
    want, sc = worked_out(test, model, reach, full_train, trees,
                          setting_text(reach.tuned_["params"]))
    want.update(stability.environment())
    check("every number of the five rankers is the one worked out here: "
          "PR-AUC ties averaged, precision@10 and nDCG@20 over tie orders",
          {k: on[k] for k in CONTEST_KEYS}, want)
    check("and on this fixture R ranks differently from path length: it "
          "reads the export flag too",
          (on["reach"] != round(evaluate(sc, "path", LABEL,
                                         ties=True)["pr_auc"], 4),
           bool(sc.groupby("public_depth")["reach"].nunique().max() > 1)),
          (True, True))
    p = sc["path"].to_numpy(float)
    order = np.argsort(-sc["path_model"].to_numpy(float), kind="mergesort")
    check("path_model only reorders path length's ties: down its order, "
          "path never rises", bool((np.diff(p[order]) <= 1e-12).all()), True)

    held = stability.one_split(df, CUT, LABEL, FEATS, "lambdarank",
                               "holdout", "binary", "fixed", contest=True)
    held_off = stability.one_split(df, CUT, LABEL, FEATS, "lambdarank",
                                   "holdout", "binary", "fixed")
    train, valid, _ = split_valid(full_train)
    train = train.sort_values(stability.GROUP)
    valid = valid.sort_values(stability.GROUP)
    m_old = fit_model(train, valid, FEATS, LABEL, "lambdarank")
    r_old = fit_model(train, valid, REACHABILITY, LABEL, "lambdarank")
    want, _ = worked_out(test, m_old, r_old, train,
                         int(getattr(r_old, "best_iteration_", None) or 600),
                         setting_text(FIXED_PARAMS))
    want.update(stability.environment())
    check("under --stopping holdout R is fitted the old way, the model's "
          "way, and every number is again the one worked out here",
          {k: held[k] for k in CONTEST_KEYS}, want)
    check("and the model's numbers are the same with item 15 on or off",
          norm({k: v for k, v in held.items() if k not in CONTEST_KEYS}),
          norm(held_off))


# -------------------------------------------------- 4. stability.py, whole

def case_script(tmp: pathlib.Path) -> None:
    print("\n4. stability.py END TO END")
    (tmp / "data").mkdir()
    labelled().to_csv(tmp / "data" / "labelled.csv", index=False)
    p = subprocess.run([sys.executable, str(ROOT / "ml/features/build.py")],
                       cwd=tmp, capture_output=True, text=True, timeout=900)
    check("build.py exits cleanly", p.returncode, 0)
    if p.returncode:
        print(p.stdout[-1500:] + p.stderr[-1500:])
        return
    at = "2026-04-01,2026-05-15"
    p = subprocess.run([sys.executable, str(ROOT / "ml/model/stability.py"),
                        "--at", at, "--contest"], cwd=tmp,
                       capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    check("stability.py --at --contest exits cleanly", p.returncode, 0)
    if p.returncode:
        print(out[-2500:])
        return
    f = tmp / "data" / "stability_label_alias_cv_tuned_at.csv"
    t = pd.read_csv(f, dtype={"cut": str, "commit": str})
    check("a sweep of the shipped model with --contest writes item 15's "
          "columns at every date",
          [k for k in CONTEST_KEYS if k not in t or t[k].isna().any()], [])
    check("and prints none of their numbers, only where they are",
          ("none are printed here" in out,
           "on the same test halves" in out, "then the model" in out),
          (True, False, False))
    p = subprocess.run([sys.executable, str(ROOT / "ml/model/stability.py"),
                        "--at", at, "--tuning", "fixed", "--contest"],
                       cwd=tmp, capture_output=True, text=True, timeout=1800)
    g = pd.read_csv(tmp / "data" / "stability_label_alias_cv_at.csv")
    check("a sweep run another way (fixed trees) fits no R and writes none, "
          "asked or not",
          (p.returncode, [k for k in CONTEST_KEYS if k in g]), (0, []))
    p = subprocess.run([sys.executable, str(ROOT / "ml/model/stability.py"),
                        "--at", at], cwd=tmp, capture_output=True,
                       text=True, timeout=1800)
    g = pd.read_csv(f)
    check("and a sweep of the shipped model without --contest writes none: "
          "the verdict is in (§36.2)",
          (p.returncode, [k for k in CONTEST_KEYS if k in g],
           "item 15" in p.stdout), (0, [], False))
    built = prepare(pd.read_csv(tmp / "data" / "features.csv"))
    plain = stability.run_label(built, LABEL, FEATS, "lambdarank", "cv",
                                at.split(","), relevance="binary",
                                tuning="cv")
    shared = [c for c in plain.columns if c in t.columns]
    differ = []
    for c in shared:
        a, b = t[c].reset_index(drop=True), plain[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and \
                pd.api.types.is_numeric_dtype(b):
            same = np.allclose(a.astype(float), b.astype(float),
                               rtol=0, atol=1e-12, equal_nan=True)
        else:
            same = a.astype(str).equals(b.astype(str))
        if not same:
            differ.append(c)
    check("the model's columns are a run without item 15's, cell for cell",
          (len(shared) > 30, differ), (True, []))


# ------------------------------------------------------------- 5. the rule

def synthetic(pr: dict, nd: dict) -> pd.DataFrame:
    """The rule's tables: each ranker's PR-AUC and nDCG@20, the same at
    every date unless a list is given. Only the rule reads these."""
    V = load("ship_verdict")
    t = pd.DataFrame({"cut": DATES})
    for k, (p, n, p10) in V.COLS.items():
        t[p] = pr.get(k, 0.20)
        t[n] = nd.get(k, 0.50)
        t[p10] = 0.25
    return t.set_index("cut")


def case_rule() -> None:
    print("\n5. THE RULE: THE ONE TEST AT ITS EDGES, EVERY BRANCH")
    V = load("ship_verdict")

    def b(pr=None, nd=None, x="R", y="P"):
        return V.beats(synthetic(pr or {}, nd or {}), x, y)["beats"]

    check("PR-AUC 1.3x at 7 of 7, nDCG@20 level: beats", b({"R": 0.26}),
          True)
    check("nDCG@20 higher at 6 of 7, PR-AUC level: beats",
          b(nd={"R": [0.52] * 6 + [0.48]}), True)
    check("ahead on PR-AUC but behind on nDCG@20 at 6 of 7: does not beat",
          b({"R": 0.26}, {"R": [0.48] * 6 + [0.52]}), False)
    check("ahead at 6 of 7 but a median of 1.24x: does not beat",
          b({"R": [0.248] * 6 + [0.18]}), False)
    check("a median of exactly 1.25x, as stored: beats", b({"R": 0.25}),
          True)
    check("nDCG@20 higher at only 5 of 7: does not beat",
          b(nd={"R": [0.52] * 5 + [0.48] * 2}), False)
    check("everything level: neither beats the other",
          (b(), b(x="P", y="R")), (False, False))

    def ships(pr=None, nd=None):
        v = V.decide(synthetic(pr or {}, nd or {}))
        return v["ships"], [(x, y) for x, y, _ in v["steps"]]

    check("nothing beats path length: P, with no order claimed in its ties",
          ships(), ("P", [("R", "P"), ("M", "P"), ("P->M", "P")]))
    check("R beats P and M does not: R, and its tie order asked next",
          ships({"R": 0.26}), ("R", [("R", "P"), ("M", "P"), ("R->M", "R")]))
    check("only M beats P: M, once R is set against M, and no tie order "
          "asked", ships({"M": 0.26}),
          ("M", [("R", "P"), ("M", "P"), ("R", "M")]))
    check("only M beats P, but the simpler R beats M: P, with its tie "
          "order asked",
          ships({"R": 0.24, "M": 0.26},
                {"M": [0.48] * 5 + [0.52] * 2, "R": [0.49] * 5 + [0.53] * 2}),
          ("P", [("R", "P"), ("M", "P"), ("R", "M"), ("P->M", "P")]))
    check("both beat P and M beats R: M",
          ships({"R": 0.26, "M": 0.34}),
          ("M", [("R", "P"), ("M", "P"), ("M", "R")]))
    check("both beat P and M does not beat R: R",
          ships({"R": 0.26, "M": 0.26}),
          ("R", [("R", "P"), ("M", "P"), ("M", "R"), ("R->M", "R")]))
    check("R beats P within upgrades while trailing it on PR-AUC (0.82x), "
          "and M beats R but not P: R, never M",
          ships({"R": 0.164, "M": 0.2146},
                {"R": [0.52] * 6 + [0.48], "M": 0.50})[0], "R")
    check("P stands, its ties ordered by the model higher on nDCG@20 at 6 "
          "of 7: P->M", ships(nd={"P->M": [0.52] * 6 + [0.48]})[0], "P->M")
    check("at only 5 of 7: P", ships(nd={"P->M": [0.52] * 5 + [0.48] * 2}
                                      )[0], "P")
    check("the tie order is judged on nDCG@20 alone: P->M at half P's "
          "PR-AUC still ships when its nDCG@20 is higher at 7 of 7",
          ships({"P->M": 0.10}, {"P->M": 0.53})[0], "P->M")
    check("R stands and its ties ordered by the model win: R->M",
          ships({"R": 0.26}, {"R->M": 0.53})[0], "R->M")
    check("the tie order is set against what stands, not against M",
          ships(nd={"P->M": 0.51, "M": [0.52] * 5 + [0.49] * 2})[0], "P->M")


# ------------------------------------------------------- 6. the verdict

def sweep(**over) -> pd.DataFrame:
    """A sweep at the seven dates measured on §34.2's data (the verdict's
    REFERENCE), every ranker level with path length unless `over` says
    otherwise: column -> a value, or one per date."""
    V = load("ship_verdict")
    ref = pd.DataFrame(V.REFERENCE).T
    t = pd.DataFrame({
        "cut": DATES, "floor": 0.05, "skipped": False,
        "holdout_from": str(HOLDOUT_START.date()), "relevance": "binary",
        "tuning": "cv", "label": LABEL, "objective": "lambdarank",
        "test_rows": ref["test_rows"].astype(int).to_list(),
        "test_pos": ref["test_pos"].astype(int).to_list(),
        "pr_auc": ref["pr_auc_ties"].to_list(), "lift_vs_pop": 3.0})
    for c in ("pr_auc_ties", "path", "ndcg_20", "path_ndcg_20", "p_at_10",
              "path_p_at_10"):
        t[c] = ref[c].to_list()
    for k, (p, n, p10) in V.COLS.items():
        if p not in t:
            t[p] = t["path"]
        t[n] = t["path_ndcg_20"]
        t[p10] = t["path_p_at_10"]
    t = t.assign(reach_trees=12, reach_setting="7/0.05/30", reach_levels=9,
                 test_hash="abc123abc123", lightgbm="4.6.0", sklearn="1.6.1",
                 pandas="2.2.3", numpy="2.1.0", python="3.12.7", cpus=10,
                 omp_threads="unset", commit="0123abc")
    for col, value in over.items():
        t[col] = value
    return t


def verdict_on(tmp: pathlib.Path, t: pd.DataFrame | None) -> tuple[int, str]:
    f = tmp / "sweep.csv"
    if t is not None:
        t.to_csv(f, index=False)
    elif f.exists():
        f.unlink()
    p = subprocess.run([sys.executable, str(ROOT / "scripts/ship_verdict.py"),
                        "sweep.csv"], cwd=tmp, capture_output=True,
                       text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def case_files(tmp: pathlib.Path) -> None:
    print("\n6. THE VERDICT, ON FILES")
    tmp = tmp / "verdict"
    tmp.mkdir()
    good = sweep()
    code, out = verdict_on(tmp, good)
    written = hashlib.sha256((tmp / "sweep.csv").read_bytes()).hexdigest()
    check("a sweep on §34.2's data, every ranker level with path: P ships",
          (code, "SHIPS: P, path length alone." in out,
           "§34.2's data: the same test rows" in out,
           "its numbers are §34.2's" in out), (0, True, True, True))
    check("and it prints what the sweep ran on, and that P->M's and "
          "R->M's PR-AUC is not ruled on",
          ("LightGBM 4.6.0" in out, "OMP_NUM_THREADS unset" in out,
           "commit 0123abc" in out, "P->M and R->M not ruled on" in out),
          (True, True, True, True))
    check("nothing is written: the folder holds only the sweep, unchanged",
          {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in tmp.iterdir()}, {"sweep.csv": written})
    digits = ["012345678901", "12345678e123"] + [f"{i}00000000000"
                                                 for i in range(1, 6)]
    code, out = verdict_on(tmp, sweep(test_hash=digits))
    check("fingerprints that all read like numbers are printed as written",
          (code, "Test halves: " + ", ".join(digits) in out), (0, True))

    code, out = verdict_on(tmp, sweep(reach=[round(x * 1.3, 4)
                                             for x in good["path"]]))
    check("R 1.3x path's PR-AUC at every date: R ships",
          (code, "SHIPS: R, the two reachability features." in out),
          (0, True))
    code, out = verdict_on(tmp, sweep(path_model_ndcg_20_e=[
        round(x + 0.02, 4) for x in good["path_ndcg_20_e"]]))
    check("path's ties ordered by the model, higher on nDCG@20 at every "
          "date: P->M ships",
          (code, "SHIPS: P->M, path length, then the model." in out),
          (0, True))
    moved = good.copy()
    moved.loc[3, "pr_auc_ties"] = 0.2133
    code, out = verdict_on(tmp, moved)
    check("the refitted model 0.01 from §34.2 at one date: ruled on, and "
          "the difference printed",
          (code, "SHIPS:" in out,
           "NOT §34.2's at\n    2026-01-18 pr_auc_ties 0.2133 (§34.2: "
           "0.2033)" in out, "§34.1's list keeps §34.2's verdict" in out),
          (0, True, True, True))
    for name, t, says in (
            ("path's PR-AUC 0.0001 from §34.2 at one date",
             good.assign(path=[x + 0.0001 if i == 1 else x for i, x in
                               enumerate(good["path"])]),
             "2025-10-06 path:"),
            ("path's fixed-order precision@10 0.0001 off at one date",
             good.assign(path_p_at_10=[x + 0.0001 if i == 6 else x for i, x
                                       in enumerate(good["path_p_at_10"])]),
             "2026-05-04 path_p_at_10"),
            ("path's fixed-order nDCG@20 0.0001 off at one date",
             good.assign(path_ndcg_20=[x + 0.0001 if i == 2 else x for i, x
                                       in enumerate(good["path_ndcg_20"])]),
             "2025-12-03 path_ndcg_20"),
            ("one test row more at one date",
             good.assign(test_rows=[x + 1 if i == 0 else x for i, x in
                                    enumerate(good["test_rows"])]),
             "2025-08-07 test_rows"),
            ("one positive more at one date",
             good.assign(test_pos=[x + 1 if i == 5 else x for i, x in
                                   enumerate(good["test_pos"])]),
             "2026-04-02 test_pos")):
        code, out = verdict_on(tmp, t)
        check(f"refused, not §34.2's data: {name}",
              (code != 0, "not measured on §34.2's data" in out, says in out,
               "SHIPS" in out), (True, True, True, False))
    for name, t, says in (
            ("a sweep from before item 15",
             good.drop(columns=CONTEST_KEYS), "written before item 15"),
            ("no test_pos column", good.drop(columns=["test_pos"]),
             "no test_pos column"),
            ("a gap in R's nDCG@20",
             good.assign(reach_ndcg_20_e=[np.nan] + list(
                 good["reach_ndcg_20_e"])[1:]), "empty value"),
            ("a gap in R's setting",
             good.assign(reach_setting=[np.nan] + ["7/0.05/30"] * 6),
             "empty value"),
            ("another label's sweep", good.assign(label="label"),
             "label is label, not label_alias"),
            ("a sweep that skipped a date",
             good.assign(skipped=[False] * 6 + [True]), "skipped 2026-05-04"),
            ("a sweep at other dates",
             good.assign(cut=DATES[:6] + ["2026-05-11"]),
             "not cut at the seven dates")):
        code, out = verdict_on(tmp, t)
        check(f"refused: {name}", (code != 0, says in out, "SHIPS" in out),
              (True, True, False))
    code, out = verdict_on(tmp, None)
    check("refused: no sweep at all, naming the sweep kept aside that the "
          "rule was applied to",
          (code != 0, "stability_label_alias_cv_tuned_at-20261011.csv" in out),
          (True, True))


# ------------------------------------------------------- 7. the constants

def case_constants() -> None:
    print("\n7. THE CONSTANTS")
    V = load("ship_verdict")
    P = load("path_verdict")
    check("the dates are §23.5's, and the measures and margins §34.1's own "
          "functions", (V.DATES, V.across.__module__, V.within.__module__,
                        P.NEED, P.MARGIN),
          (DATES, "path_verdict", "path_verdict", 6, 1.25))
    check("the five rankers, simplest first, each read on PR-AUC with ties "
          "averaged and the tie-averaged nDCG@20 and precision@10",
          (list(V.COLS), V.COLS["P"], V.COLS["M"], V.COLS["P->M"]),
          (["P", "R", "M", "P->M", "R->M"],
           ("path", "path_ndcg_20_e", "path_p_at_10_e"),
           ("pr_auc_ties", "ndcg_20_e", "p_at_10_e"),
           ("path_model", "path_model_ndcg_20_e", "path_model_p_at_10_e")))
    notes = (ROOT / "docs" / "NOTES.md").read_text()
    sec = notes.split("### 34.2 ")[1].split("\n### ")[0].split("\n## ")[0]
    rows = re.findall(r"^\| (\d{4}-\d\d-\d\d) \| ([\d.]+) \| ([\d.]+) \| "
                      r"([\d.]+) \| [\d.]+x \| ([\d.]+) \| ([\d.]+) \| "
                      r"([\d.]+) \| ([\d.]+) \|$", sec, re.M)
    recorded = {d: dict(pr_auc_ties=float(m), path=float(p),
                        ndcg_20=float(mn), path_ndcg_20=float(pn),
                        p_at_10=float(m10), path_p_at_10=float(p10))
                for d, _, m, p, mn, pn, m10, p10 in rows}
    floors = {d: float(fl) for d, fl, *_ in rows}
    kept = {d: {k: r[k] for k in recorded.get(d, {})}
            for d, r in V.REFERENCE.items()}
    check("REFERENCE is §34.2's table in NOTES, number for number",
          (len(rows), kept), (7, recorded))
    check("its positives over its test rows are §34.2's floor at every date",
          {d: round(r["test_pos"] / r["test_rows"], 4)
           for d, r in V.REFERENCE.items()}, floors)
    sec = notes.split("### 33.3 ")[1].split("\n### ")[0].split("\n## ")[0]
    after = dict(re.findall(r"^\| (\d{4}-\d\d-\d\d) \| [^|]+\| [^|]+\| "
                            r"[\d,]+ -> ([\d,]+) \|", sec, re.M))
    check("and its test rows are §33.3's",
          {d: r["test_rows"] for d, r in V.REFERENCE.items()},
          {d: int(v.replace(",", "")) for d, v in after.items()})


def main() -> None:
    case_then_by()
    case_expected()
    case_fingerprint()
    df = dev()
    case_spread(df)
    case_row(df)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-ship-"))
    try:
        case_script(tmp)
        case_rule()
        case_files(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    case_constants()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The five rankers are fitted and scored as "
          "§36.1 says,\nand the verdict follows its rule, refusing any "
          "sweep not measured on §34.2's data.")


if __name__ == "__main__":
    main()
