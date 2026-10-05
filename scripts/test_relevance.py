"""
Item 3 of the fix list (F2): is graded relevance what it says it is, and
is binary still exactly what it was?

    python scripts/test_relevance.py

No network and no real data; about a minute. It takes the labelled
fixture test_holdout.py builds, gives its positives usage counts, and runs
the real code on it, in a temp directory. Nothing in your data/ or
artifacts/ is read or touched.

WHY THIS FILE EXISTS. Graded relevance can look done while doing nothing.
A model trained on the 0/1 label under a graded name, a sweep stamped
graded whose fits were binary, a train.py that quotes a binary sweep for
a graded model: each of those still prints numbers, and none of them are
the numbers of a graded model.

Seven cases:

  1. The grades: 0 for a negative, then 1 to 4 by the count the label is
     built from, with edges at 1 | 2-6 | 7-19 | 20+. A positive is never
     0 and a negative never above it, whatever the count says.
  2. What LightGBM is handed. Graded: every fit, each CV fold with its
     validation rows and then the refit, gets the grades and the gains
     [0, 1, 3, 7, 15]. Binary: the 0/1 label and the gains [0, 1], as
     before item 3. The binary objective refuses graded, and a graded run
     with no count column stops before the CV loop, instead of failing
     every fold quietly and coming back with the 600-tree cap.
  3. Graded does what it is for. In this fixture the short names are the
     changes many packages use, and the 0/1 label does not depend on the
     name. Graded training orders positives of different grades better
     than binary does, and scores higher on nDCG@20 with graded gains.
  4. nDCG@20 with graded gains: the plain number when every gain is the
     label; a worked example that plain nDCG@20 scores 1.0 both ways
     round and graded does not; the same upgrades judged either way.
  5. The scripts, end to end. stability.py writes a _graded file, every
     row stamped, with the same cuts skipped as binary. train.py names a
     graded model apart, and quotes a sweep only if its models were
     taught the same way, by the stamp and not just the file name.
     ablate.py and final_eval.py run graded and say so, and final_eval's
     number is what a graded fit on dev rows alone gives.
  6. Every fit each script makes under --relevance graded, watched at
     LightGBM itself: train.py and stability.py with both stopping rules,
     ablate.py with its CV tree count, and final_eval.py. A path that
     forgot to pass the relevance on would still print believable
     numbers; here it cannot hide.
  7. The rule in scripts/item3_relevance.py, at every edge: exactly 0.25x
     below binary's worst case still holds and a little more does not;
     6 of 7 dates wins and 5 does not; a tie is not a win; one date lost
     to popularity fails; more than 0.25x above binary's worst case wins
     alone, exactly 0.25x does not; a date not measured means no verdict.
"""

import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.build import (BOOLEAN, CATEGORICAL, NUMERIC,  # noqa: E402
                               add_features, drop_version_strings)
from ml.holdout import (GROUP, HOLDOUT_START, assert_no_holdout,  # noqa: E402
                        drop_holdout)
from ml.model import train as T  # noqa: E402
from ml.model.metrics import ndcg_at_k  # noqa: E402

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


def run(script: str, cwd: pathlib.Path, *args: str):
    p = subprocess.run([sys.executable, str(ROOT / script), *args],
                       cwd=cwd, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def probe(cwd: pathlib.Path, code: str) -> str:
    p = subprocess.run([sys.executable, "-c",
                        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
                        + code],
                       cwd=cwd, capture_output=True, text=True, timeout=600)
    return p.stdout + p.stderr


# ------------------------------------------------------------------ fixture

def labelled() -> pd.DataFrame:
    """test_holdout.py's labelled.csv, with usage counts. A positive with a
    short name (fn0 to fn9) is used by 20 to 79 packages, grade 4; the
    rest mostly by one, some by 2-6 or 7-19. The 0/1 label is drawn
    without looking at the name, so only graded training has a reason to
    care about it."""
    df = load("test_holdout").make_labelled(seed=0)
    rng = np.random.default_rng(1)
    short = (df["name_length"] <= 3).to_numpy()
    r = rng.random(len(df))
    few = np.where(r < 0.75, 1, np.where(r < 0.9, rng.integers(2, 7, len(df)),
                                         rng.integers(7, 20, len(df))))
    count = np.where(short, rng.integers(20, 80, len(df)), few)
    for lab, col in T.COUNT_OF.items():
        df[col] = np.where(df[lab] == 1, count, 0)
    return df


def dev_features(lab: pd.DataFrame) -> pd.DataFrame:
    """The dev rows as the model sees them. The direct fits in this file
    use these only; the holdout has no business here either."""
    dev = drop_holdout(add_features(drop_version_strings(lab)))
    assert_no_holdout(dev, "test_relevance.py")
    return T.prepare(dev).sort_values(GROUP)


# -------------------------------------------------------------------- cases

def case_grades() -> None:
    print("\n1. THE GRADES")
    n = [0, 1, 2, 6, 7, 19, 20, 500, 1, 0, 3]
    lab = [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
    df = pd.DataFrame({"alias_user_count": n, "label_alias": lab})
    check("edges: 1 -> 1, 2 and 6 -> 2, 7 and 19 -> 3, 20 and 500 -> 4",
          T.grades(df, LABEL).tolist()[:8], [0, 1, 2, 2, 3, 3, 4, 4])
    check("the label decides which rows count: a negative with a count "
          "is 0, a positive with none is 1",
          T.grades(df, LABEL).tolist()[8:], [0, 1, 0])
    df = pd.DataFrame({"user_count": [0, 30], "scoped_user_count": [30, 0],
                       "label": [0, 1], "label_scoped": [1, 0]})
    check("each label reads its own count",
          (T.grades(df, "label").tolist(),
           T.grades(df, "label_scoped").tolist()), ([0, 4], [4, 0]))
    check("the gains are 2**grade - 1",
          T.graded_gain(pd.DataFrame({"alias_user_count": [0, 1, 3, 10, 50],
                                      "label_alias": [0, 1, 1, 1, 1]}),
                        LABEL).tolist(), [0, 1, 3, 7, 15])


def spy_fits(fn):
    """Run fn with LGBMRanker.fit wrapped, and return what every fit was
    handed: its label_gain, its y and its validation y."""
    seen = []
    real = lgb.LGBMRanker.fit

    def spy(self, X, y, *args, **kw):
        ev = kw.get("eval_set")
        seen.append({"gain": list(self.get_params()["label_gain"]),
                     "y": np.asarray(y).copy(),
                     "eval_y": None if not ev else np.asarray(ev[0][1]).copy()})
        return real(self, X, y, *args, **kw)

    lgb.LGBMRanker.fit = spy
    try:
        fn()
    finally:
        lgb.LGBMRanker.fit = real
    return seen


def case_handed(dev: pd.DataFrame) -> None:
    print("\n2. WHAT LIGHTGBM IS HANDED")
    train = dev[pd.to_datetime(dev["released_at"]) <= "2026-05-15"]

    seen = spy_fits(lambda: T.fit_cv(train, FEATS, LABEL, "lambdarank",
                                     relevance="graded"))
    folds = [s for s in seen if s["eval_y"] is not None]
    check(f"graded: every fit ({len(seen)}) gets the gains [0, 1, 3, 7, 15]",
          all(s["gain"] == T.GRADE_GAIN for s in seen) and len(seen) > 1,
          True)
    check("every CV fold trains AND validates on grades, not 0/1",
          (len(folds) > 0,
           all(s["y"].max() > 1 and s["eval_y"].max() > 1 for s in folds)),
          (True, True))
    check("the refit is handed exactly the grades of the training rows",
          seen[-1]["y"].tolist(),
          T.grades(train.sort_values(GROUP), LABEL).tolist())

    seen = spy_fits(lambda: T.fit_cv(train, FEATS, LABEL, "lambdarank"))
    check("binary: every fit gets the 0/1 label and gains [0, 1], as "
          "before item 3",
          (all(s["gain"] == [0, 1] and set(s["y"]) <= {0, 1} for s in seen),
           seen[-1]["y"].tolist()),
          (True, train.sort_values(GROUP)[LABEL].tolist()))

    for what, call in (
            ("fit_fixed", lambda: T.fit_fixed(train, FEATS, LABEL, 5,
                                              "binary", "graded")),
            ("fit_cv", lambda: T.fit_cv(train, FEATS, LABEL, "binary",
                                        relevance="graded"))):
        try:
            call()
            got = "ran"
        except ValueError as e:
            got = "ranking target" in str(e)
        check(f"{what}: the binary objective refuses graded", got, True)

    try:
        got = T.cv_tree_count(train.drop(columns="alias_user_count"), FEATS,
                              LABEL, "lambdarank", relevance="graded")
    except ValueError as e:
        got = "alias_user_count" in str(e)
    check("no count column: cv_tree_count stops, it does not return the "
          "600-tree cap", got, True)


def concordance(frame: pd.DataFrame) -> float:
    """Of the pairs of positives in one upgrade whose grades differ, the
    share the model scores the higher grade above."""
    good = total = 0
    for _, g in frame.groupby(GROUP, sort=False):
        p = g[g[LABEL] == 1]
        gr, sc = T.grades(p, LABEL).to_numpy(), p["s"].to_numpy()
        hi, lo = np.nonzero(gr[:, None] > gr[None, :])
        total += len(hi)
        good += int((sc[hi] > sc[lo]).sum())
    return good / total if total else float("nan")


def case_purpose(dev: pd.DataFrame) -> None:
    print("\n3. GRADED DOES WHAT IT IS FOR")
    when = pd.to_datetime(dev["released_at"])
    train, test = dev[when <= "2026-05-15"], dev[when > "2026-05-15"]
    got = {}
    for rel in ("binary", "graded"):
        m = T.fit_fixed(train, FEATS, LABEL, 60, "lambdarank", rel)
        s = test.assign(s=T.score_with(m, test, FEATS),
                        _gain=T.graded_gain(test, LABEL))
        got[rel] = (concordance(s),
                    ndcg_at_k(s, "s", LABEL, 20, gain="_gain"))
    (bc, bn), (gc, gn) = got["binary"], got["graded"]
    check(f"it orders positives of different grades better: {gc:.2f} of "
          f"pairs against binary's {bc:.2f}", gc > bc, True)
    check(f"nDCG@20 with graded gains is higher: {gn:.4f} against "
          f"{bn:.4f}", gn > bn, True)


def case_metric(dev: pd.DataFrame) -> None:
    print("\n4. nDCG@20 WITH GRADED GAINS")
    rng = np.random.default_rng(3)
    s = dev.assign(s=rng.random(len(dev)),
                   _one=dev[LABEL].astype(float))
    check("with the label as the gain, it is plain nDCG@20, exactly",
          ndcg_at_k(s, "s", LABEL, 20, gain="_one"),
          ndcg_at_k(s, "s", LABEL, 20))

    # One upgrade of 21 changes: A used by 30 packages (gain 15), B by one
    # (gain 1), 19 nobody uses. A then B is ideal. B then A scores
    # (1 + 15/log2 3) / (15 + 1/log2 3) = 0.6694 with gains, and both
    # orders score 1.0 without them.
    up = pd.DataFrame({"package": "p", "version_from": "1", "version_to": "2",
                       LABEL: [1, 1] + [0] * 19,
                       "alias_user_count": [30, 1] + [0] * 19,
                       "a_first": [2.0, 1.0] + [0.0] * 19,
                       "b_first": [1.0, 2.0] + [0.0] * 19})
    up["_gain"] = T.graded_gain(up, LABEL)
    want = (1 + 15 / np.log2(3)) / (15 + 1 / np.log2(3))
    check("A then B scores 1.0, B then A 0.6694; plain nDCG@20 cannot tell",
          (round(ndcg_at_k(up, "a_first", LABEL, 20, gain="_gain"), 4),
           round(ndcg_at_k(up, "b_first", LABEL, 20, gain="_gain"), 4),
           ndcg_at_k(up, "a_first", LABEL, 20),
           ndcg_at_k(up, "b_first", LABEL, 20)),
          (1.0, round(want, 4), 1.0, 1.0))
    small = up.iloc[1:].assign(version_to="3")
    check("an upgrade of 20 changes is not judged either way",
          (ndcg_at_k(small, "a_first", LABEL, 20, gain="_gain"),
           ndcg_at_k(small, "a_first", LABEL, 20)), (0.0, 0.0))


def case_scripts(tmp: pathlib.Path, lab: pd.DataFrame) -> None:
    print("\n5. THE SCRIPTS, END TO END")
    (tmp / "data").mkdir()
    lab.to_csv(tmp / "data" / "labelled.csv", index=False)
    code, out = run("ml/features/build.py", tmp)
    check("build.py exits cleanly", code, 0)
    if code:
        print(out[-1500:])
        return
    data = tmp / "data"
    dates = "2026-04-01,2026-05-15,2026-07-20"

    # Pinned to --tuning fixed: what this case tests is the same either
    # way, and its fixtures are named for the fixed sweep. Tuning has its
    # own tests (test_tuning.py).
    code, out = run("ml/model/stability.py", tmp, "--at", dates,
                    "--relevance", "graded", "--tuning", "fixed")
    graded = data / "stability_label_alias_cv_graded_at.csv"
    check("stability.py --relevance graded writes the _graded file",
          (code, graded.exists()), (0, True))
    code, out = run("ml/model/stability.py", tmp, "--at", dates,
                    "--tuning", "fixed")
    binary = data / "stability_label_alias_cv_at.csv"
    if not (graded.exists() and binary.exists()):
        print(out[-1500:])
        return
    g, b = pd.read_csv(graded), pd.read_csv(binary)
    check("every row says what its model was taught",
          (set(g["relevance"]), set(b["relevance"])), ({"graded"}, {"binary"}))
    check("the same cuts are skipped, on the same test positives",
          (g["skipped"].tolist(), g["test_pos"].tolist()),
          (b["skipped"].tolist(), b["test_pos"].tolist()))
    check("the late cut is one of them, so that was tested",
          bool(g["skipped"].iloc[-1]), True)
    fitted = ~g["skipped"].astype(bool)
    check("both carry nDCG@20 with graded gains, and the models differ",
          (bool(g.loc[fitted, "ndcg_20_graded"].notna().all()),
           bool(b.loc[fitted, "ndcg_20_graded"].notna().all()),
           g.loc[fitted, "pr_auc"].tolist() != b.loc[fitted, "pr_auc"].tolist()),
          (True, True, True))
    check("and it is not plain nDCG@20 under another name",
          bool((b.loc[fitted, "ndcg_20_graded"]
                != b.loc[fitted, "ndcg_20"]).any()), True)

    # Two quantile sweeps train.py could quote, told apart by their size.
    stamp = str(HOLDOUT_START.date())
    sweep = pd.DataFrame({"cut": ["2026-04-01", "2026-05-01", "2026-06-01"],
                          "skipped": False, "lift_vs_pop": [2.5, 2.7, 2.9],
                          "beats_pop": True, "holdout_from": stamp})
    sweep.assign(relevance="graded").to_csv(
        data / "stability_label_alias_cv_graded.csv", index=False)
    sweep.head(2).assign(relevance="binary").to_csv(
        data / "stability_label_alias_cv.csv", index=False)
    metrics = tmp / "artifacts" / "metrics.json"

    code, out = run("ml/model/train.py", tmp, "--relevance", "graded",
                    "--tuning", "fixed")
    run_row = pd.read_json(metrics, typ="series") if metrics.exists() else {}
    check("train.py --relevance graded runs and names its model apart",
          (code, run_row.get("version")),
          (0, "lambdarank-label_alias-graded"))
    check("its notes say relevance=graded, and quote the graded sweep only",
          ("relevance=graded" in run_row.get("notes", ""),
           "across 3 cut dates" in run_row.get("notes", "")), (True, True))
    check("it prints the grades it trained on and nDCG@20 with graded gains",
          ("training positives by grade" in out,
           "nDCG@20 with graded gains" in out), (True, True))

    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    run_row = pd.read_json(metrics, typ="series")
    check("binary train.py keeps its old name and quotes the binary sweep",
          (code, run_row["version"], "across 2 cut dates" in run_row["notes"]),
          (0, "lambdarank-label_alias", True))

    # The stamp, not the name: a graded sweep saved under the binary name.
    sweep.head(2).assign(relevance="graded").to_csv(
        data / "stability_label_alias_cv.csv", index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    run_row = pd.read_json(metrics, typ="series")
    check("a sweep stamped graded is not quoted for a binary model, "
          "whatever its file is called",
          (code, "across" in run_row["notes"], "graded relevance" in out),
          (0, False, True))
    # A sweep from before item 3 has no stamp, and was binary.
    sweep.head(2).to_csv(data / "stability_label_alias_cv.csv", index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    run_row = pd.read_json(metrics, typ="series")
    check("an unstamped sweep from before item 3 is quoted as binary",
          (code, "across 2 cut dates" in run_row["notes"]), (0, True))

    code, out = run("ml/model/train.py", tmp, "--objective", "binary",
                    "--relevance", "graded")
    check("--objective binary --relevance graded is refused, and says why",
          (code != 0, "ranking target" in out, "Traceback" in out),
          (True, True, False))

    features = data / "features.csv"
    good = features.read_text()
    pd.read_csv(features).drop(columns="alias_user_count").to_csv(
        features, index=False)
    for script in ("ml/model/train.py", "ml/model/stability.py",
                   "ml/model/ablate.py"):
        code, out = run(script, tmp, "--relevance", "graded")
        check(f"{script} refuses graded with no count column, naming it",
              (code != 0, "alias_user_count" in out, "Traceback" in out),
              (True, True, False))
    features.write_text(good)

    code, out = run("ml/model/ablate.py", tmp, "--relevance", "graded",
                    "--trees", "5", "--tuning", "fixed")
    check("ablate.py --relevance graded runs and writes its own file",
          (code, "relevance graded" in out,
           (data / "ablation_label_alias_graded.csv").exists()),
          (0, True, True))

    code, out = run("ml/model/final_eval.py", tmp, "--unseal",
                    "--relevance", "graded", "--tuning", "fixed")
    ledger = data / "holdout_ledger.csv"
    row = pd.read_csv(ledger).iloc[-1] if ledger.exists() else {}
    check("final_eval.py --relevance graded runs and records it",
          (code, row.get("objective"), "relevance graded" in out,
           "nDCG@20 graded" in out),
          (0, "lambdarank+graded", True, True))
    expected = probe(tmp, (
        "import pandas as pd\n"
        "from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC\n"
        "from ml.holdout import GROUP\n"
        "from ml.model.metrics import evaluate\n"
        "from ml.model.train import fit_cv, prepare, score_with\n"
        "t = {c: str for c in GROUP}\n"
        "d = pd.read_csv('data/features.csv', dtype=t).assign(_p=0)\n"
        "h = pd.read_csv('data/holdout.csv', dtype=t).assign(_p=1)\n"
        "b = prepare(pd.concat([d, h], ignore_index=True))\n"
        "d = b[b._p == 0].drop(columns='_p').sort_values(GROUP)\n"
        "h = b[b._p == 1].drop(columns='_p').sort_values(GROUP).copy()\n"
        "f = NUMERIC + BOOLEAN + CATEGORICAL\n"
        "for rel in ('graded', 'binary'):\n"
        "    m, _, _ = fit_cv(d, f, 'label_alias', 'lambdarank',"
        " relevance=rel)\n"
        "    h['s'] = score_with(m, h, f)\n"
        "    print(rel.upper(), evaluate(h, 's', 'label_alias')['pr_auc'])\n"))
    want = {k: float(v) for k, v in
            re.findall(r"(GRADED|BINARY) ([0-9.]+)", expected)}
    got = float(row["pr_auc"]) if len(row) else None
    check("its PR-AUC is a graded fit on dev rows alone, and not the "
          "binary one", (len(want) == 2 and got is not None
                         and abs(got - want["GRADED"]) < 1e-6
                         and abs(got - want["BINARY"]) > 1e-6), True)
    if len(want) != 2:
        print(expected[-1500:])


def spied_script(script: str, cwd: pathlib.Path, *args: str):
    """Run one of the real scripts' main() in this process, under the fit
    spy, and return (every fit it made, its exit code, its output)."""
    import contextlib
    import io
    import os
    import runpy
    out, code = io.StringIO(), 0
    here, argv = os.getcwd(), sys.argv
    os.chdir(cwd)
    sys.argv = [script, *args]

    def go():
        nonlocal code
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(str(ROOT / script), run_name="__main__")
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            out.write(str(e.code))
    try:
        seen = spy_fits(go)
    finally:
        os.chdir(here)
        sys.argv = argv
    return seen, code, out.getvalue()


def case_every_fit(tmp: pathlib.Path) -> None:
    print("\n6. EVERY FIT THE SCRIPTS MAKE UNDER GRADED IS HANDED GRADES")
    # Watched at LightGBM itself, so a path that forgets to pass the
    # relevance on (a stopping mode, the CV count, the ablation refits)
    # shows up here even where its numbers would look plausible.
    if not (tmp / "data" / "features.csv").exists():
        check("the fixture was built", False, True)
        return
    runs = [("train.py, CV stopping", "ml/model/train.py",
             "--stopping", "cv"),
            ("train.py, single-slice stopping", "ml/model/train.py",
             "--stopping", "holdout"),
            ("stability.py, CV stopping", "ml/model/stability.py",
             "--at", "2026-05-15"),
            ("stability.py, single-slice stopping", "ml/model/stability.py",
             "--at", "2026-05-15", "--stopping", "holdout"),
            ("ablate.py, CV tree count", "ml/model/ablate.py"),
            ("final_eval.py", "ml/model/final_eval.py", "--unseal",
             "--again", "test_relevance: every fit graded")]
    for name, script, *args in runs:
        # --tuning fixed: graded is binary-tuning-only (test_tuning.py).
        seen, code, out = spied_script(script, tmp, *args,
                                       "--relevance", "graded",
                                       "--tuning", "fixed")
        ok = (code == 0 and len(seen) > 0
              and all(s["gain"] == T.GRADE_GAIN and s["y"].max() > 1
                      and (s["eval_y"] is None or s["eval_y"].max() > 1)
                      for s in seen))
        check(f"{name}: all {len(seen)} fits graded", ok, True)
        if code:
            print(out[-1500:])
    # Control: the spy does see a binary fit when there is one.
    seen, code, _ = spied_script("ml/model/train.py", tmp)
    check("control: plain train.py is seen fitting 0/1 with gains [0, 1]",
          (code, len(seen) > 0,
           all(s["gain"] == [0, 1] and s["y"].max() == 1 for s in seen)),
          (0, True, True))


def case_rule() -> None:
    print("\n7. THE RULE, AT EVERY EDGE")
    i3 = load("item3_relevance")
    dates = i3.DATES

    def frame(lifts, ndcg, skipped=None, beats=None) -> pd.DataFrame:
        return pd.DataFrame({
            "lift_vs_pop": lifts, "ndcg_20_graded": ndcg,
            "skipped": skipped or [False] * 7,
            "beats_pop": beats or [v > 1 for v in lifts]}, index=dates)

    base = [4.42, 4.86, 5.70, 4.02, 3.01, 4.08, 2.30]
    b = frame(base, [0.5] * 7)
    up6 = [0.6] * 6 + [0.4]
    up5 = [0.6] * 5 + [0.4] * 2

    def ships(g: pd.DataFrame, bb: pd.DataFrame = b):
        return i3.verdict(bb, g)[0]

    check("worst 2.05x, exactly 0.25x below, and 6/7 on nDCG: ships",
          ships(frame(base[:6] + [2.05], up6)), True)
    check("worst 2.04x, a little more than 0.25x below: stays binary",
          ships(frame(base[:6] + [2.04], up6)), False)
    check("5/7 on nDCG and no lift win: stays binary",
          ships(frame(base, up5)), False)
    check("6 higher and one tie is 6/7; 5 higher and two ties is 5/7",
          (ships(frame(base, [0.6] * 6 + [0.5])),
           ships(frame(base, [0.6] * 5 + [0.5] * 2))), (True, False))
    check("worst 2.56x, more than 0.25x above, wins with 0/7 on nDCG",
          ships(frame(base[:6] + [2.56], [0.4] * 7)), True)
    check("worst 2.55x, exactly 0.25x above, does not",
          ships(frame(base[:6] + [2.55], [0.4] * 7)), False)
    low = frame(base[:6] + [1.10], [0.5] * 7)
    check("one date lost to popularity fails, within 0.25x or not",
          ships(frame(base[:6] + [0.95], up6), low), False)
    check("control: the same, winning that date, ships",
          ships(frame(base[:6] + [1.05], up6), low), True)
    check("a date not measured: no verdict",
          ships(frame(base, up6, skipped=[False] * 6 + [True])), None)


def main() -> None:
    lab = labelled()
    dev = dev_features(lab)
    case_grades()
    case_handed(dev)
    case_purpose(dev)
    case_metric(dev)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-relevance-"))
    try:
        case_scripts(tmp, lab)
        case_every_fit(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    case_rule()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Graded models are trained on grades, named")
    print("and quoted apart, and binary is what it was before item 3.")


if __name__ == "__main__":
    main()
