"""
Item 7, second half (F7): is tuning what it says it is, and is the
shipped model exactly what it was?

    python scripts/test_tuning.py

No network and no real data; a minute or two. It uses test_holdout.py's
labelled fixture, runs the real code on it in a temp directory, and
reads nothing in your data/ or artifacts/.

WHY THIS FILE EXISTS. Tuning can look done while doing nothing, or while
cheating. A sweep stamped tuned whose fits used the shipped setting, a
fold that stopped on nDCG@10 after all, a tree count clamped at 20 under
a tuned name, or a choice that peeked at test rows: each still prints
believable numbers.

Seven cases:

  1. The grid and the choice: 18 distinct settings, simplest first, the
     shipped one among them; the best median score wins and a tie goes
     to the simpler setting.
  2. The shipped model is what it was: --tuning fixed fits exactly the
     model a LightGBM ranker built from the literal shipped setting does,
     byte for byte, with the tree count cv_tree_count gives.
  3. Every tuned fold stops on PR-AUC, trains only on rows dated before
     its validation window, and sees no row outside train; the refit has
     no validation set and grows exactly the chosen number of trees.
  4. The choice, by hand: with the fold fits stubbed, a setting whose
     folds rank perfectly beats one that ranks nothing, its score is the
     median of PR-AUC over each fold's floor, its tree count is the
     median of its folds' with no clamp (4 trees, not 20), and the refit
     uses its setting, not the shipped one.
  5. Refusals: graded relevance, single-slice stopping and an unknown
     tuning are refused before anything is fitted, and say why.
  6. The scripts end to end: stability.py writes a _tuned file, every
     row stamped and each fitted cut naming a setting from the grid;
     train.py names a tuned model apart and quotes a sweep only if its
     trees were sized the same way, by the stamp and not the file name;
     with no option it is now the tuned model (§30.2); ablate.py holds
     one tuned setting and count for every run, in its own file;
     final_eval.py records a tuned opening and its setting.
  7. The rule in scripts/item7_tuning.py at every edge, and the count
     of validation upgrades with no positive that F7's diagnosis rests
     on.
"""

import importlib.util
import itertools
import pathlib
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
from ml.holdout import GROUP, HOLDOUT_START, drop_holdout  # noqa: E402
from ml.model import train as T  # noqa: E402

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
                       cwd=cwd, capture_output=True, text=True, timeout=1200)
    return p.returncode, p.stdout + p.stderr


def dev_train() -> pd.DataFrame:
    """The fixture's dev rows released by 15 May: a training half."""
    dev = T.prepare(drop_holdout(add_features(drop_version_strings(
        load("test_holdout").make_labelled())))).sort_values(GROUP)
    return dev[pd.to_datetime(dev["released_at"]) <= "2026-05-15"]


def spy(fn):
    """Run fn with both LightGBM fits wrapped; return what each was given."""
    seen = []
    real = {c: c.fit for c in (lgb.LGBMRanker, lgb.LGBMClassifier)}

    def make(cls):
        def wrapped(self, X, y, *args, **kw):
            ev = kw.get("eval_set")
            seen.append({"index": X.index.copy(),
                         "eval_index": (None if not ev
                                        else ev[0][0].index.copy()),
                         "metric": self.get_params().get("metric"),
                         "n_estimators": self.get_params()["n_estimators"],
                         "params": {k: self.get_params()[k]
                                    for k in T.FIXED_PARAMS}})
            return real[cls](self, X, y, *args, **kw)
        return wrapped

    for c in real:
        c.fit = make(c)
    try:
        out = fn()
    finally:
        for c, f in real.items():
            c.fit = f
    return seen, out


# -------------------------------------------------------------------- cases

def case_grid() -> None:
    print("\n1. THE GRID AND THE CHOICE")
    shapes = [tuple(sorted(g.items())) for g in T.GRID]
    check("18 settings, all different", (len(T.GRID), len(set(shapes))),
          (18, 18))
    check("the shipped setting is one of them", T.FIXED_PARAMS in T.GRID,
          True)
    check("and it is the shipped setting: 31 leaves, 0.05, smallest leaf 30",
          T.FIXED_PARAMS, {"num_leaves": 31, "learning_rate": 0.05,
                           "min_child_samples": 30})
    check("simplest first: 7 leaves with smallest leaf 100",
          (T.GRID[0]["num_leaves"], T.GRID[0]["min_child_samples"]),
          (7, 100))
    rows = [{"order": 0, "score": 2.0}, {"order": 1, "score": 3.0},
            {"order": 2, "score": 3.0}]
    check("the best score wins, and a tie goes to the earlier (simpler) one",
          T.choose_setting(rows)["order"], 1)
    check("a setting written as leaves/learning rate/smallest leaf",
          T.setting_text(T.FIXED_PARAMS), "31/0.05/30")
    check("tuned ships, by §30.1's rule (§30.2): the default is cv",
          T.SHIPPED_TUNING, "cv")


def case_shipped(train: pd.DataFrame) -> None:
    print("\n2. THE SHIPPED MODEL IS WHAT IT WAS")
    model, n, _ = T.fit_cv(train, FEATS, LABEL, "lambdarank")
    n_cv, _ = T.cv_tree_count(train, FEATS, LABEL, "lambdarank")
    literal = lgb.LGBMRanker(
        objective="lambdarank", label_gain=[0, 1], n_estimators=n_cv,
        learning_rate=0.05, num_leaves=31, min_child_samples=30,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        random_state=0, verbose=-1)
    t = train.sort_values(GROUP)
    literal.fit(t[FEATS], t[LABEL], group=T.group_sizes(t))
    check("--tuning fixed is the literal shipped ranker, byte for byte",
          model.booster_.model_to_string()
          == literal.booster_.model_to_string(), True)
    check("with cv_tree_count's tree count, clamp included", n, n_cv)
    check("and it carries no tuning record", hasattr(model, "tuned_"), False)


def case_folds(train: pd.DataFrame) -> None:
    print("\n3. EVERY TUNED FOLD STOPS ON PR-AUC, INSIDE TRAIN, BEFORE ITS "
          "WINDOW")
    small = [T.GRID[0], T.FIXED_PARAMS]
    real = T.GRID
    T.GRID = small
    try:
        seen, (model, n, folds) = spy(lambda: T.fit_cv(
            train, FEATS, LABEL, "lambdarank", tuning="cv"))
    finally:
        T.GRID = real
    n_folds = len(list(T.cv_folds(train, LABEL)))
    fold_fits, refit = seen[:-1], seen[-1]
    check(f"one fit per setting per fold, then one refit "
          f"({len(small)} x {n_folds} + 1)", len(seen),
          len(small) * n_folds + 1)
    check("every fold fit stops on average_precision, over a window",
          all(f["metric"] == "average_precision"
              and f["eval_index"] is not None for f in fold_fits), True)
    when = pd.to_datetime(train["released_at"])
    check("every fold trains only on rows dated before its window",
          all(when.loc[f["index"]].max() < when.loc[f["eval_index"]].min()
              for f in fold_fits), True)
    check("and no fit sees a row outside train",
          all(set(f["index"]) <= set(train.index)
              and (f["eval_index"] is None
                   or set(f["eval_index"]) <= set(train.index))
              for f in seen), True)
    check("each setting tried is the grid's",
          sorted({tuple(sorted(f["params"].items())) for f in fold_fits}),
          sorted(tuple(sorted(g.items())) for g in small))
    check("the refit has no window and grows exactly the chosen trees",
          (refit["eval_index"], refit["n_estimators"],
           model.booster_.num_trees()), (None, n, n))
    check("with the chosen setting",
          refit["params"], model.tuned_["params"])


def case_by_hand(train: pd.DataFrame) -> None:
    print("\n4. THE CHOICE, BY HAND")

    class Stub:
        """A fold's model: it stopped at `best` trees, and scores the
        window it was fitted for either by its labels (a perfect ranking)
        or all alike (no ranking at all)."""
        def __init__(self, best, labels):
            self.best_iteration_, self.labels = best, labels

        def predict(self, frame):
            return self.labels.loc[frame.index].to_numpy(float)

    # b is not the shipped setting, so a refit that ignored the choice
    # would show.
    a, b = T.GRID[0], T.GRID[1]
    # Setting a's four folds, then b's; again for fit_cv's own cv_tune.
    trees = itertools.cycle([7, 8, 9, 10, 3, 9, 4, 5])

    def stub(sub, val, feats, label, objective, params):
        labels = (val[LABEL].astype(float) if params == b
                  else pd.Series(0.0, index=val.index))
        return Stub(next(trees), labels)

    real, real_grid = T.fit_stopping_on_ap, T.GRID
    T.fit_stopping_on_ap, T.GRID = stub, [a, b]
    try:
        params, n, fold_trees, table = T.cv_tune(train, FEATS, LABEL,
                                                 "lambdarank")
        model, n2, _ = T.fit_cv(train, FEATS, LABEL, "lambdarank",
                                tuning="cv")
    finally:
        T.fit_stopping_on_ap, T.GRID = real, real_grid
    folds = list(T.cv_folds(train, LABEL))
    check("the fixture has four folds, so the stub's trees line up",
          len(folds), 4)
    floors = [val[LABEL].mean() for _, val in folds]
    check("a setting whose folds rank perfectly beats one that ranks "
          "nothing", params, b)
    by = {r["order"]: r for r in table}
    check("ranking nothing scores PR-AUC = floor, so 1.0x the floor",
          round(by[0]["score"], 9), 1.0)
    check("ranking perfectly scores the median of 1 / each fold's floor",
          round(by[1]["score"], 9), round(float(np.median(
              [1 / f for f in floors])), 9))
    check("its trees: the median of its folds' [3, 9, 4, 5], 4, not "
          "clamped to 20", (fold_trees, n), ([3, 9, 4, 5], 4))
    check("and fit_cv refits with exactly those 4 trees",
          (n2, model.booster_.num_trees()), (4, 4))
    check("in the chosen setting, not the shipped one",
          {k: model.get_params()[k] for k in T.FIXED_PARAMS}, b)


def case_refusals(train: pd.DataFrame) -> None:
    print("\n5. REFUSALS")
    check("graded relevance is refused, and says why",
          "binary relevance only" in (T.tuning_problem("cv", "graded")
                                      or ""), True)
    check("single-slice stopping is refused",
          "--stopping cv" in (T.tuning_problem("cv", "binary", "holdout")
                              or ""), True)
    check("an unknown tuning is refused",
          T.tuning_problem("random") is not None, True)
    check("and the shipped way needs nothing",
          T.tuning_problem("fixed", "graded", "holdout"), None)
    seen, raised = spy(lambda: _raises(lambda: T.fit_cv(
        train, FEATS, LABEL, "lambdarank", relevance="graded",
        tuning="cv")))
    check("fit_cv refuses graded + cv before fitting anything",
          (raised, len(seen)), (True, 0))


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def case_scripts(tmp: pathlib.Path) -> None:
    print("\n6. THE SCRIPTS, END TO END")
    (tmp / "data").mkdir()
    load("test_holdout").make_labelled().to_csv(
        tmp / "data" / "labelled.csv", index=False)
    code, out = run("ml/features/build.py", tmp)
    check("build.py exits cleanly", code, 0)
    if code:
        print(out[-1500:])
        return
    data = tmp / "data"
    dates = "2026-04-01,2026-05-15"

    code, out = run("ml/model/stability.py", tmp, "--at", dates,
                    "--tuning", "cv")
    tuned = data / "stability_label_alias_cv_tuned_at.csv"
    check("stability.py --tuning cv writes the _tuned file",
          (code, tuned.exists()), (0, True))
    code, out2 = run("ml/model/stability.py", tmp, "--at", dates,
                     "--tuning", "fixed")
    fixed = data / "stability_label_alias_cv_at.csv"
    if not (tuned.exists() and fixed.exists()):
        print((out + out2)[-1500:])
        return
    tu, fx = pd.read_csv(tuned), pd.read_csv(fixed)
    fitted = ~tu["skipped"].astype(bool)
    grid = {T.setting_text(g) for g in T.GRID}
    check("every row says how its trees were sized",
          (set(tu["tuning"]), set(fx["tuning"])), ({"cv"}, {"fixed"}))
    check("each tuned cut names a setting from the grid; fixed ones the "
          "shipped one",
          (bool(fitted.any()) and set(tu.loc[fitted, "setting"]) <= grid,
           set(fx.loc[~fx["skipped"].astype(bool), "setting"])),
          (True, {"31/0.05/30"}))
    check("the same cuts are skipped either way",
          tu["skipped"].tolist(), fx["skipped"].tolist())
    check("the report prints each cut's setting", "setting" in out, True)

    stamp = str(HOLDOUT_START.date())
    sweep = pd.DataFrame({"cut": ["2026-04-01", "2026-05-01", "2026-06-01"],
                          "skipped": False, "lift_vs_pop": [2.5, 2.7, 2.9],
                          "beats_pop": True, "holdout_from": stamp,
                          "relevance": "binary"})
    sweep.assign(tuning="cv").to_csv(
        data / "stability_label_alias_cv_tuned.csv", index=False)
    sweep.head(2).assign(tuning="fixed").to_csv(
        data / "stability_label_alias_cv.csv", index=False)
    metrics = tmp / "artifacts" / "metrics.json"

    code, out = run("ml/model/train.py", tmp, "--tuning", "cv")
    row = pd.read_json(metrics, typ="series") if metrics.exists() else {}
    check("train.py --tuning cv runs and names its model apart",
          (code, row.get("version")), (0, "lambdarank-label_alias-tuned"))
    notes = row.get("notes", "")
    check("its notes say tuning=cv and the setting, and quote the tuned "
          "sweep only",
          ("tuning=cv" in notes, "setting=" in notes,
           "across 3 cut dates" in notes), (True, True, True))
    check("it prints the settings it tried and the one it chose",
          ("tuned on the same folds" in out, "chosen" in out), (True, True))

    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    row = pd.read_json(metrics, typ="series")
    check("fixed train.py keeps its old name and quotes the fixed sweep",
          (code, row["version"], "tuning=fixed" in row["notes"],
           "across 2 cut dates" in row["notes"]),
          (0, "lambdarank-label_alias", True, True))

    sweep.head(2).assign(tuning="cv").to_csv(
        data / "stability_label_alias_cv.csv", index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    row = pd.read_json(metrics, typ="series")
    check("a sweep stamped cv is not quoted for a fixed model, whatever "
          "its file is called",
          (code, "across" in row["notes"], "-tuned models" in out),
          (0, False, True))
    sweep.head(2).drop(columns="relevance").to_csv(
        data / "stability_label_alias_cv.csv", index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    row = pd.read_json(metrics, typ="series")
    check("an unstamped sweep from before item 7 is quoted as fixed",
          (code, "across 2 cut dates" in row["notes"]), (0, True))

    code, out = run("ml/model/train.py", tmp)
    row = pd.read_json(metrics, typ="series") if metrics.exists() else {}
    check("with no option, train.py is the tuned model now, and quotes the "
          "tuned sweep",
          (code, row.get("version"), "tuning=cv" in row.get("notes", ""),
           "across 3 cut dates" in row.get("notes", "")),
          (0, "lambdarank-label_alias-tuned", True, True))

    code, out = run("ml/model/ablate.py", tmp)
    check("ablate.py, by default, holds one tuned setting and count for "
          "every run, in its own file",
          (code, "chosen once on all features" in out, "tuning cv" in out,
           (data / "ablation_label_alias_tuned.csv").exists()),
          (0, True, True, True))
    code, out = run("ml/model/ablate.py", tmp, "--tuning", "fixed")
    check("and --tuning fixed is the ablation as before, in the old file",
          (code, "chosen once" in out, "tuning fixed" in out,
           (data / "ablation_label_alias.csv").exists()),
          (0, False, True, True))

    # This fixture has no count column, so graded is refused for that
    # first; either reason is a refusal that says why.
    for args, why in ((("--tuning", "cv", "--relevance", "graded"),
                       ("binary relevance only", "alias_user_count")),
                      (("--tuning", "cv", "--stopping", "holdout"),
                       ("--stopping cv",))):
        code, out = run("ml/model/train.py", tmp, *args)
        check(f"train.py {' '.join(args)} is refused, and says why",
              (code != 0, any(w in out for w in why), "Traceback" in out),
              (True, True, False))

    code, out = run("ml/model/final_eval.py", tmp, "--unseal", "--tuning",
                    "cv")
    ledger = data / "holdout_ledger.csv"
    led = pd.read_csv(ledger).iloc[-1] if ledger.exists() else {}
    check("final_eval.py --tuning cv runs and records a tuned opening",
          (code, led.get("objective"), "setting=" in str(led.get("cv_folds")),
           "tuned on the dev folds" in out),
          (0, "lambdarank+tuned", True, True))


def case_rule() -> None:
    print("\n7. THE RULE, AT EVERY EDGE")
    I = load("item7_tuning")
    fixed = list(I.SHIPPED)          # worst 2.30x, median 4.08x

    def ships(tuned, beats=7):
        return I.verdict(fixed, tuned, beats)[0]

    at_both = [4.42, 3.58, 5.70, 4.02, 3.01, 3.58, 2.05]
    check("the dates and shipped lifts are §23.5's and §24.2's",
          (len(I.DATES), I.SHIPPED[0], I.SHIPPED[-1], I.WORST_MARGIN,
           I.MEDIAN_MARGIN), (7, 4.42, 2.30, 0.25, 0.50))
    check("worst exactly 0.25x below and median exactly 0.50x below: ships",
          (min(at_both), float(np.median(at_both)), ships(at_both)),
          (2.05, 3.58, True))
    check("worst 0.26x below: fixed stays",
          ships([x if x != 2.05 else 2.04 for x in at_both]), False)
    check("median 0.51x below: fixed stays",
          ships([4.42, 3.57, 5.70, 4.02, 3.01, 3.57, 2.05]), False)
    check("loses to popularity at one date: fixed stays",
          ships(list(I.SHIPPED), beats=6), False)
    check("the same lifts, beating popularity everywhere: ships",
          ships(list(I.SHIPPED)), True)
    try:
        I.verdict(fixed, fixed[:6], 7)
        refused = False
    except ValueError:
        refused = True
    check("six lifts against seven is refused", refused, True)

    df = pd.DataFrame({
        "package": [f"p{i}" for i in range(10) for _ in range(2)],
        "version_from": "1", "version_to": "2",
        "released_at": np.repeat(pd.date_range("2026-01-01", periods=10,
                                               freq="7D").astype(str), 2),
        LABEL: [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]})
    got = I.empty_share(df)
    want = []
    for _, val in T.cv_folds(df.sort_values(GROUP), LABEL):
        per = val.groupby(GROUP)[LABEL].sum()
        want.append((len(per), int((per == 0).sum())))
    check("upgrades with no positive are counted per fold window",
          (got, bool(got) and all(e <= n for n, e in got)), (want, True))


def main() -> None:
    train = dev_train()
    case_grid()
    case_shipped(train)
    case_folds(train)
    case_by_hand(train)
    case_refusals(train)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-tuning-"))
    try:
        case_scripts(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    case_rule()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Tuned folds stop on PR-AUC inside train, the")
    print("tree count is the data's with no clamp, and the shipped model is")
    print("byte for byte what it was.")


if __name__ == "__main__":
    main()
