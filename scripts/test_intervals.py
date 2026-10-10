"""
Item 5 of the fix list (F11, F12): are the intervals what they say?

    python scripts/test_intervals.py

No network and no real data; about a minute. Nothing in your data/ or
artifacts/ is read or touched.

WHY THIS FILE EXISTS. An interval is only as honest as its resampling.
Resample rows instead of upgrades and it comes out narrow and confident;
forget the weights in one of the four numbers and that number's interval
describes a different quantity from the one printed beside it. Neither
failure shows in the output: both print a plausible range.

Six cases:

  1. One resample, recomputed by hand. With a single draw, each interval
     is that draw's value, which is rebuilt here the slow way: the drawn
     upgrades stacked as many times as they were drawn for PR-AUC and the
     lift, and each rankable upgrade's own precision@10 and nDCG@20
     averaged with the same counts.
  2. The band is the middle 95%, and a resample with nothing to measure
     is left out rather than counted as zero.
  3. Whole upgrades are resampled. When how well the model does is
     decided upgrade by upgrade, the interval is wide, against a control
     that resamples rows and comes out three times narrower.
  4. On a realistic fixture every interval holds its own point estimate,
     and the same seed gives the same intervals.
  5. The scripts: train.py prints the intervals and writes them into
     model_run's notes with the change and upgrade counts; stability.py
     writes each cut's lift interval and counts the cuts where it clears
     1.0x; train.py quotes that count, and says nothing of it for a sweep
     from before item 5; final_eval.py prints them for the holdout.
  6. The count "clear of 1.0x" reads each cut's lower bound, not its
     point: a cut whose lift is above 1.0x with an interval that reaches
     below it is not counted.
"""

import contextlib
import importlib.util
import io
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

from ml.features.build import (BOOLEAN, CATEGORICAL, NUMERIC,  # noqa: E402
                               add_features, drop_version_strings)
from ml.holdout import GROUP, HOLDOUT_START, drop_holdout  # noqa: E402
from ml.model import metrics as M  # noqa: E402
from ml.model import stability  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.train import fit_fixed, prepare, score_with  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []
LABEL = "label_alias"
FEATS = NUMERIC + BOOLEAN + CATEGORICAL


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def close(a: float, b: float) -> bool:
    return bool(abs(a - b) < 1e-9)


def labelled() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(
        "test_holdout", ROOT / "scripts" / "test_holdout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_labelled()


def scored_test() -> pd.DataFrame:
    """A realistic scored test half: the fixture's dev rows, a model fitted
    on those before 15 May, scored with the baselines on the rest."""
    dev = prepare(drop_holdout(add_features(drop_version_strings(
        labelled())))).sort_values(GROUP)
    # drop_holdout above is the tripwire's other form; no holdout row is
    # scored here.
    when = pd.to_datetime(dev["released_at"])
    train, test = dev[when <= "2026-05-15"], dev[when > "2026-05-15"]
    model = fit_fixed(train, FEATS, LABEL, 20)
    test = test.assign(model=score_with(model, test, FEATS))
    return add_baseline_scores(train, test, LABEL)


def case_by_hand(sc: pd.DataFrame) -> None:
    print("\n1. ONE RESAMPLE, RECOMPUTED BY HAND")
    ok = []
    for seed in (1, 2, 3):
        got = M.intervals(sc, LABEL, draws=1, seed=seed)
        df = sc.reset_index(drop=True)
        key = df[M.PAIR].astype(str).agg("\x1f".join, axis=1)
        codes, uniq = pd.factorize(key)
        counts = np.random.default_rng(seed).multinomial(
            len(uniq), np.full(len(uniq), 1 / len(uniq)), size=1)[0]
        rows = np.repeat(np.arange(len(df)), counts[codes])
        stacked = df.iloc[rows]
        y = stacked[LABEL].to_numpy(int)
        ap_m = average_precision_score(y, stacked["model"])
        ap_p = average_precision_score(y, stacked["popularity"])
        p10, n20, w10, w20 = [], [], [], []
        for c, (_, g) in zip(pd.factorize(key)[1],
                             df.groupby(key, sort=False)):
            n = counts[list(uniq).index(c)]
            if g[LABEL].sum() and len(g) > 10:
                p10.append(M.precision_at_k(g, "model", LABEL, 10))
                w10.append(n)
            if g[LABEL].sum() and len(g) > 20:
                n20.append(M.ndcg_at_k(g, "model", LABEL, 20))
                w20.append(n)
        want = {"pr_auc": ap_m, "baseline_pr_auc": ap_p, "lift": ap_m / ap_p,
                "precision_at_10": np.average(p10, weights=w10),
                "ndcg_at_20": np.average(n20, weights=w20)}
        ok.append(all(close(got[k][0], v) and close(got[k][1], v)
                      for k, v in want.items()))
    check("PR-AUC, popularity, lift, precision@10 and nDCG@20 are each the "
          "drawn upgrades' own value (3 seeds)", ok, [True, True, True])


def case_band() -> None:
    print("\n2. THE MIDDLE 95%")
    check("0..1000: the band is 25 to 975",
          M._band(np.arange(1001)), (25.0, 975.0))
    check("a resample with nothing to measure is left out, not read as 0",
          M._band(np.array([np.nan] * 500 + list(np.arange(1001)))),
          (25.0, 975.0))


def case_upgrades() -> None:
    print("\n3. WHOLE UPGRADES ARE RESAMPLED")
    # 10 upgrades of 50 changes, 10 of them used in each. In five the
    # model puts the used ones on top; in the other five, at the bottom.
    # How well it does is decided upgrade by upgrade, which is what real
    # releases look like, and a row-by-row resample cannot see.
    rng = np.random.default_rng(0)
    parts = []
    for p in range(10):
        y = np.r_[np.ones(10, int), np.zeros(40, int)]
        s = (y if p < 5 else 1 - y) * 2 + rng.random(50)
        parts.append(pd.DataFrame({"package": f"p{p}", "version_from": "1",
                                   "version_to": "2", LABEL: y, "model": s,
                                   "popularity": rng.random(50)}))
    df = pd.concat(parts, ignore_index=True)
    lo, hi = M.intervals(df, LABEL)["pr_auc"]
    # The control: the same number of draws, of rows instead of upgrades.
    rng = np.random.default_rng(0)
    y, s = df[LABEL].to_numpy(), df["model"].to_numpy()
    naive = []
    for _ in range(M.DRAWS):
        i = rng.integers(0, len(df), len(df))
        naive.append(average_precision_score(y[i], s[i]))
    nlo, nhi = M._band(np.array(naive))
    check(f"when upgrades decide it, it shows: {lo:.2f} to {hi:.2f}, against "
          f"{nlo:.2f} to {nhi:.2f} resampling rows",
          (hi - lo) > 2 * (nhi - nlo), True)


def case_brackets(sc: pd.DataFrame) -> None:
    print("\n4. EVERY INTERVAL HOLDS ITS OWN NUMBER")
    ci = M.intervals(sc, LABEL)
    m, p = M.evaluate(sc, "model", LABEL), M.evaluate(sc, "popularity", LABEL)
    point = {"pr_auc": m["pr_auc"], "baseline_pr_auc": p["pr_auc"],
             "lift": m["pr_auc"] / p["pr_auc"],
             "precision_at_10": m["precision_at_10"],
             "ndcg_at_20": m["ndcg_at_20"]}
    check("PR-AUC, popularity, lift, precision@10, nDCG@20 each inside",
          {k: ci[k][0] <= v <= ci[k][1] for k, v in point.items()},
          {k: True for k in point})
    check("the upgrades counted are the test half's",
          ci["upgrades"], len(sc.drop_duplicates(GROUP)))
    check("the same seed gives the same intervals",
          M.intervals(sc, LABEL) == ci, True)


def run(script: str, cwd: pathlib.Path, *args: str):
    p = subprocess.run([sys.executable, str(ROOT / script), *args],
                       cwd=cwd, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def case_scripts(tmp: pathlib.Path) -> None:
    print("\n5. THE SCRIPTS")
    (tmp / "data").mkdir()
    labelled().to_csv(tmp / "data" / "labelled.csv", index=False)
    code, out = run("ml/features/build.py", tmp)
    check("build.py exits cleanly", code, 0)
    if code:
        return
    data = tmp / "data"

    # Pinned to --tuning fixed: what this case tests is the same either
    # way, and its fixtures are named for the fixed sweep. Tuning has its
    # own tests (test_tuning.py).
    code, out = run("ml/model/stability.py", tmp, "--at",
                    "2026-04-01,2026-05-15", "--tuning", "fixed")
    t = pd.read_csv(data / "stability_label_alias_cv_at.csv")
    check("stability.py writes each cut's lift interval, around its lift, "
          "and its changes and upgrades",
          (code, bool((t["lift_lo"] <= t["lift_vs_pop"]).all()
                      and (t["lift_vs_pop"] <= t["lift_hi"]).all()),
           bool((t["test_changes"] <= t["test_rows"]).all()),
           t["test_upgrades"].gt(0).all()), (0, True, True, True))
    check("and says at how many cuts that interval clears 1.0x",
          "and by more than its own 95% interval (lower end above 1.0x) at"
          in out, True)

    stamp = str(HOLDOUT_START.date())
    sweep = pd.DataFrame({"cut": ["2026-04-01", "2026-05-01"],
                          "skipped": False, "lift_vs_pop": [1.8, 1.3],
                          "beats_pop": True, "holdout_from": stamp})
    sweep.assign(lift_lo=[1.4, 0.9], lift_hi=[2.3, 1.8]).to_csv(
        data / "stability_label_alias_cv.csv", index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    notes = pd.read_json(tmp / "artifacts" / "metrics.json",
                         typ="series")["notes"]
    check("train.py prints the intervals, with the test half's changes "
          "and upgrades",
          (code, "95% intervals, from 2,000 resamples" in out,
           "rows are" in out and "changes (a symbol in one upgrade) in" in out),
          (0, True, True))
    check("model_run's notes carry them: pr_auc_95, lift_95, the counts",
          all(k in notes for k in ("pr_auc_95=", "lift_95=",
                                   "test_changes=", "test_upgrades=")), True)
    check("and quote the sweep's count by its lower bounds: 1 of 2, not 2",
          "lift's 95% interval above 1.0x at 1/2" in notes, True)
    sweep.to_csv(data / "stability_label_alias_cv.csv", index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    notes = pd.read_json(tmp / "artifacts" / "metrics.json",
                         typ="series")["notes"]
    check("a sweep from before item 5 is quoted, and nothing is said of "
          "intervals it does not have",
          (code, "across 2 cut dates" in notes,
           "interval above 1.0x" in notes), (0, True, False))

    code, out = run("ml/model/final_eval.py", tmp, "--unseal",
                    "--tuning", "fixed")
    found = re.search(r"model\s+PR-AUC\s+([0-9.]+)\s+95% interval "
                      r"([0-9.]+) to ([0-9.]+)", out)
    lift = re.search(r"lift over popularity\s+([0-9.]+)x\s+95% interval "
                     r"([0-9.]+) to ([0-9.]+)", out)
    around = (bool(found and lift) and all(
        float(m.group(2)) <= float(m.group(1)) <= float(m.group(3))
        for m in (found, lift)))
    check("final_eval.py prints real intervals around its PR-AUC and lift, "
          "and puts them in the NOTES block",
          (code, around, "intervals resample upgrades" in out),
          (0, True, True))


def case_report() -> None:
    print("\n6. 'CLEAR OF 1.0x' READS THE LOWER BOUND")
    t = pd.DataFrame({"cut": ["2026-01-01", "2026-02-01"], "q": [0.6, 0.7],
                      "test_rows": 100, "test_pos": 40, "floor": 0.1,
                      "trees": 20, "rankable10": 12, "pr_auc": 0.3,
                      "popularity": 0.2, "lift_vs_pop": [1.5, 1.2],
                      "lift_lo": [1.1, 0.95], "lift_hi": [2.0, 1.6],
                      "p_at_10": 0.2, "ndcg_20": 0.5, "beats_pop": True,
                      "beats_semver": True, "skipped": False})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        stability.report(t, LABEL)
    text = out.getvalue()
    check("both beat popularity; one clears its interval",
          ("beats popularity at 2/2" in text,
           "(lower end above 1.0x) at 1/2" in text), (True, True))
    check("each cut's interval is printed beside its lift",
          ("1.10-2.00" in text, "0.95-1.60" in text), (True, True))


def main() -> None:
    sc = scored_test()
    case_by_hand(sc)
    case_band()
    case_upgrades()
    case_brackets(sc)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-intervals-"))
    try:
        case_scripts(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    case_report()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Each interval resamples whole upgrades, and")
    print("describes the number printed beside it.")


if __name__ == "__main__":
    main()
