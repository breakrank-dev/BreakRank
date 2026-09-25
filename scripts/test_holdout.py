"""
Is the frozen holdout really out of reach of every experiment?

    python scripts/test_holdout.py

No network and no real data. It writes a synthetic labelled.csv into a
temp directory, runs the real scripts against it, and checks that holdout
rows reach nothing except final_eval.py --unseal. About a minute, because
it fits small models; nothing in your data/ or artifacts/ is read or
touched.

WHY THIS FILE EXISTS.

The freeze (ml/holdout.py) is only as good as its weakest entry point.
Before it, stability.py ignored the `split` column and cut every row at
its own dates, so its late cuts tested on exactly the releases that are
now the holdout. A freeze that fixed build.py and forgot stability.py
would have looked finished and leaked on every run.

Eight cases:

  1. The boundary: inclusive, a date, decided per version pair, and not
     fooled by a second date format or a timezone.
  2. split_off() loses no row and puts no pair on both sides.
  3. build.py seals it: holdout rows land in holdout.csv, never in
     features.csv. The pair list saved on the first build is never
     rewritten, and later builds say how the set has moved against it.
     An older dataset with nothing to hold out still builds.
  4. Every evaluation script refuses a features.csv built before the
     freeze, and every one of them still runs on a clean file. The
     refusal is recognised by its message, not just a non-zero exit, so
     a script that crashes for some other reason cannot pass as
     "refused". fold_effect.py, which drops the holdout instead of
     refusing it, is run through its drop path too.
  5. Every script that imports the metrics or the fitting functions
     calls the tripwire. A text scan: it cannot see a notebook, but it
     does catch a new script that forgets.
  6. The measurability gates in ml/holdout.py match stability.py's.
  7. final_eval.py is sealed by default, records the first opening,
     refuses a second without a reason, and its number is what training
     on dev rows ALONE gives, recomputed independently here.
  8. train.py will not quote a stability file written before the freeze
     into model_run, and ml/db.py still scores holdout rows so the site
     keeps its newest releases, including when holdout.csv is empty or
     its versions look like numbers. Serving is not evaluating.

Every guard above was also broken on purpose, one at a time, to watch
this file fail: the tripwire removed from stability.py and from ablate.py,
the pre-freeze build.py restored, the stale-file check in train.py
loosened, the boundary made exclusive, the ledger check in final_eval.py
disabled, final_eval.py made to train on dev + holdout, db.py's holdout
read switched off, db.py made to concatenate an empty holdout, and the
second-format date parse removed. Each of the ten breaks turned at least
one line to FAIL. A test that has never been seen to fail has not yet
shown it can.
"""

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

from ml.features.build import add_features  # noqa: E402
from ml.holdout import (GROUP, HOLDOUT_START, MIN_POSITIVES,  # noqa: E402
                        MIN_RANKABLE_PAIRS, holdout_mask, split_off)

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def run(script: str, cwd: pathlib.Path, *args: str):
    """Run one of the real scripts with cwd as its working directory, so
    its relative data/ and artifacts/ paths point at the fixture."""
    p = subprocess.run([sys.executable, str(ROOT / script), *args],
                       cwd=cwd, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def probe(cwd: pathlib.Path, code: str) -> str:
    """Run a few lines of Python inside the fixture, repo importable."""
    p = subprocess.run([sys.executable, "-c",
                        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
                        + code],
                       cwd=cwd, capture_output=True, text=True, timeout=600)
    return p.stdout + p.stderr


# ------------------------------------------------------------------ fixture

KINDS = ["OBJECT_REMOVED", "PARAMETER_MOVED", "PARAMETER_REMOVED",
         "ATTRIBUTE_CHANGED_VALUE", "PARAMETER_ADDED_REQUIRED",
         "RETURN_CHANGED_TYPE"]


def make_labelled(seed: int = 0, n_packages: int = 60) -> pd.DataFrame:
    """A labelled.csv-shaped frame: 60 packages, 7 upgrades each, released
    between January and late September 2026, so roughly a fifth of the
    pairs fall on or after 2026-08-04. Labels lean on public_depth,
    is_top_level and is_private, so a model has something to learn and
    the metrics are not all zero."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_packages):
        pkg = f"pkg{p:02d}"
        rank = p + 1
        day = pd.Timestamp("2026-01-05") + pd.Timedelta(
            days=int(rng.integers(0, 90)))
        major, minor, patch = 1, int(rng.integers(0, 5)), 0
        versions = [f"{major}.{minor}.{patch}"]
        dates = []
        for _ in range(7):
            step = rng.choice(["patch", "minor", "major"], p=[.6, .3, .1])
            if step == "patch":
                patch += 1
            elif step == "minor":
                minor, patch = minor + 1, 0
            else:
                major, minor, patch = major + 1, 0, 0
            versions.append(f"{major}.{minor}.{patch}")
            day = min(day + pd.Timedelta(days=int(rng.integers(8, 40))),
                      pd.Timestamp("2026-09-20"))
            dates.append(day)
        for vf, vt, when in zip(versions, versions[1:], dates):
            for i in range(int(rng.integers(12, 26))):
                kind = KINDS[int(rng.integers(0, len(KINDS)))]
                method = rng.random() < 0.6
                private = rng.random() < 0.15
                name = f"{'_' if private else ''}fn{i}"
                symbol = (f"{pkg}.core.Thing.{name}" if method
                          else f"{pkg}.api.{name}")
                depth = symbol.count(".")
                public_depth = int(rng.integers(1, depth + 1))
                top = public_depth == 1
                logit = (-3.4 + 1.6 * top - 0.5 * (public_depth - 1)
                         - 2.0 * private - 0.6 * method
                         + rng.normal(0, 0.8))
                label = int(rng.random() < 1 / (1 + np.exp(-logit)))
                alias = int(label or rng.random() < 0.02)
                rows.append({
                    "package": pkg, "version_from": vf, "version_to": vt,
                    "symbol": symbol, "kind": kind,
                    "sub_target": ("arg" if kind.startswith("PARAMETER")
                                   else ""),
                    "released_at": when.strftime("%Y-%m-%d"),
                    "module_depth": depth, "public_depth": public_depth,
                    "name_length": len(name), "package_rank": rank,
                    "inherited_by": 0,
                    "prior_breaks_in_module": int(rng.integers(0, 30)),
                    "was_deprecated_before": bool(rng.random() < 0.02),
                    "is_private": private, "is_dunder": False,
                    "in_dunder_all": bool(top and rng.random() < 0.5),
                    "is_top_level": top, "has_export_path": top,
                    "user_count": label * int(rng.integers(1, 9)),
                    "label": label, "label_scoped": alias,
                    "label_alias": alias,
                })
    return pd.DataFrame(rows)


def stale_features(labelled: pd.DataFrame) -> pd.DataFrame:
    """features.csv the way build.py wrote it BEFORE the freeze: every
    row, quantile split at 0.75, holdout rows sitting inside "test"."""
    df = add_features(labelled)
    when = pd.to_datetime(df["released_at"], errors="coerce")
    df["split"] = np.where(when > when.quantile(0.75), "test", "train")
    return df


# -------------------------------------------------------------------- cases

def case_boundary() -> None:
    print("\n1. THE BOUNDARY: INCLUSIVE, A DATE, DECIDED PER PAIR")
    df = pd.DataFrame({
        "package": ["a", "b", "c", "d", "e", "e"],
        "version_from": ["1.0"] * 6,
        "version_to": ["1.1"] * 6,
        "released_at": ["2026-08-03", "2026-08-04", "2026-08-05", None,
                        "2026-08-03", "2026-08-04"],
    })
    m = holdout_mask(df).tolist()
    check("the day before the boundary is dev", m[0], False)
    check("the boundary day itself is holdout", m[1], True)
    check("the day after is holdout", m[2], True)
    check("an undated pair stays out of the holdout", m[3], False)
    check("a pair whose rows disagree goes to the holdout WHOLE",
          m[4:], [True, True])

    # One column, three shapes of date. The fast parse infers a format from
    # the first value and would turn the other two into NaT, which is
    # "undated", which is dev: a post-boundary row trained on, silently.
    mixed = pd.DataFrame({
        "package": ["f", "g", "h"], "version_from": ["1"] * 3,
        "version_to": ["2"] * 3,
        "released_at": ["2026-07-01", "2026-08-10T12:00:00",
                        "2026-08-04T03:00:00+05:30"]})
    check("a second date format is parsed, not dropped to dev; "
          "timezones compare in UTC",
          holdout_mask(mixed).tolist(), [False, True, False])


def case_partition(labelled: pd.DataFrame) -> None:
    print("\n2. split_off() LOSES NOTHING AND SPLITS NO PAIR")
    dev, hold = split_off(labelled)
    check("every row lands on exactly one side",
          len(dev) + len(hold), len(labelled))
    both = dev[GROUP].drop_duplicates().merge(
        hold[GROUP].drop_duplicates(), on=GROUP)
    check("no version pair is on both sides", len(both), 0)
    dd = pd.to_datetime(dev["released_at"])
    hd = pd.to_datetime(hold["released_at"])
    check("every dev row is older than the boundary",
          bool((dd < HOLDOUT_START).all()), True)
    check("every holdout row is on or after it",
          bool((hd >= HOLDOUT_START).all()), True)
    check("the fixture puts a real share of pairs in the holdout",
          0.1 < len(hold) / len(labelled) < 0.4, True)


def case_build(tmp: pathlib.Path, labelled: pd.DataFrame) -> None:
    print("\n3. build.py SEALS THE HOLDOUT")
    code, out = run("ml/features/build.py", tmp)
    check("build.py exits cleanly", code, 0)
    if code:
        print(out[-2000:])
        return
    feats = pd.read_csv(tmp / "data" / "features.csv")
    check("features.csv holds no holdout row",
          int(holdout_mask(feats).sum()), 0)
    hold_path = tmp / "data" / "holdout.csv"
    check("holdout.csv is written", hold_path.exists(), True)
    if not hold_path.exists():
        return
    hold = pd.read_csv(hold_path)
    check("holdout.csv holds nothing else",
          bool(holdout_mask(hold).all()), True)
    check("together they are every input row, once",
          len(feats) + len(hold), len(labelled))
    check("dev rows are split train/test, holdout rows say holdout",
          (sorted(feats["split"].unique()), hold["split"].unique().tolist()),
          (["test", "train"], ["holdout"]))
    check("build.py reports the frozen holdout and the time-order count",
          ("HOLDOUT, frozen" in out, "time order:" in out), (True, True))

    # THE PAIR LIST. Written once, then every later build reports against
    # it. Simulate the two legitimate ways the set moves: a fix empties one
    # frozen pair (F1 dropping its only rows), and a retry adds a package.
    manifest = tmp / "data" / "holdout_manifest.csv"
    check("the first build saves the frozen pair list",
          (manifest.exists(), len(pd.read_csv(manifest)) if manifest.exists()
           else 0), (True, len(hold[GROUP].drop_duplicates())))
    frozen_text = manifest.read_text() if manifest.exists() else ""
    gone = hold[GROUP].drop_duplicates().iloc[0]
    is_gone = (labelled[GROUP] == gone.values).all(axis=1)
    newcomer = labelled[~is_gone].head(3).assign(
        package="latecomer", version_from="1.0.0", version_to="1.1.0",
        released_at="2026-09-21")
    moved = pd.concat([labelled[~is_gone], newcomer], ignore_index=True)
    moved.to_csv(tmp / "data" / "labelled.csv", index=False)
    code, out = run("ml/features/build.py", tmp)
    check("a later build reports what moved: 1 gone, 1 added",
          (code, "1 gone, 1 added" in out), (0, True))
    check("and leaves the frozen list exactly as it was",
          manifest.read_text() if manifest.exists() else None, frozen_text)
    labelled.to_csv(tmp / "data" / "labelled.csv", index=False)
    code, out = run("ml/features/build.py", tmp)
    check("back on the original data, the set is as frozen",
          (code, "the set is as frozen" in out), (0, True))

    # An OLDER dataset, with nothing on or after the boundary, is still a
    # valid thing to build (historical re-runs); it just has nothing to seal.
    old = tmp / "older"
    (old / "data").mkdir(parents=True)
    split_off(labelled)[0].to_csv(old / "data" / "labelled.csv", index=False)
    code, out = run("ml/features/build.py", old)
    empty = old / "data" / "holdout.csv"
    check("a dataset with no holdout builds, says so, and writes no list",
          (code, "HOLDOUT: EMPTY" in out,
           len(pd.read_csv(empty)) if empty.exists() else None,
           (old / "data" / "holdout_manifest.csv").exists()),
          (0, True, 0, False))


def case_tripwire(tmp: pathlib.Path, labelled: pd.DataFrame) -> None:
    print("\n4. EVERY EVALUATION SCRIPT REFUSES A PRE-FREEZE features.csv")
    fresh = (tmp / "data" / "features.csv").read_text()
    stale = stale_features(labelled)

    # The fixture reproduces the leak the guard exists for: under the old
    # code the latest stability cut tested on every row after its date,
    # holdout included. If this ever reads 0, the checks below prove little.
    when = pd.to_datetime(stale["released_at"])
    late_test = stale[when > when.quantile(0.85)]
    check("the stale file WOULD put holdout rows in the old q=0.85 test half",
          int(holdout_mask(late_test).sum()) > 0, True)

    scripts = {"ml/model/baselines.py": (),
               "ml/model/train.py": (),
               "ml/model/stability.py": (),
               "ml/model/ablate.py": ("--trees", "5"),
               "scripts/label_blindspot.py": ()}
    stale.to_csv(tmp / "data" / "features.csv", index=False)
    try:
        for script, args in scripts.items():
            code, out = run(script, tmp, *args)
            check(f"{script} refuses it",
                  (code != 0, "HOLDOUT ROWS" in out), (True, True))
    finally:
        (tmp / "data" / "features.csv").write_text(fresh)

    # And on the clean file every one of them runs to the end. train.py is
    # left to case 8, which needs its artifacts.
    for script, args in scripts.items():
        if script.endswith("train.py"):
            continue
        code, out = run(script, tmp, *args)
        check(f"{script} runs on the frozen features.csv",
              (code, "HOLDOUT ROWS" in out), (0, False))
        if code:
            print(out[-1500:])

    # fold_effect.py reads changes.csv, which legitimately holds the
    # holdout, so it DROPS those rows instead of refusing them.
    labels_made_later = ["label", "label_scoped", "label_alias", "user_count",
                         "is_private", "is_dunder", "has_export_path",
                         "public_depth"]
    changes = labelled.drop(columns=labels_made_later)
    changes.to_csv(tmp / "data" / "changes.csv", index=False)
    changes.to_csv(tmp / "data" / "changes-amplified.csv", index=False)
    used = labelled[labelled["user_count"] > 0]
    (used.groupby("symbol")["user_count"].max().reset_index()
     .to_csv(tmp / "data" / "usage.csv", index=False))
    code, out = run("scripts/fold_effect.py", tmp, "--cut", "2026-08-15")
    check("fold_effect.py refuses a cut inside the holdout",
          (code != 0, "inside the frozen holdout" in out), (True, True))
    code, out = run("scripts/fold_effect.py", tmp, "--cut", "2026-06-01")
    n_dev = len(split_off(labelled)[0])
    check("fold_effect.py drops the holdout and runs on dev rows only",
          (code, f"{n_dev:,} rows ->" in out), (0, True))
    if code:
        print(out[-1500:])
    for name in ("changes.csv", "changes-amplified.csv", "usage.csv"):
        (tmp / "data" / name).unlink()


def case_structure() -> None:
    print("\n5. EVERY SCRIPT THAT IMPORTS THE SCORING CODE CALLS THE TRIPWIRE")
    # "Imports the scoring code" = the metrics module or the fitting
    # functions. metrics.py defines them and this file tests them;
    # everything else that touches them must also call the guard.
    scores = re.compile(r"from ml\.model\.metrics import|"
                        r"from ml\.model import metrics|"
                        r"\b(fit_cv|fit_fixed|fit_model|cv_tree_count)\(")
    guard = re.compile(r"^\s*[^#\n]*\b(assert_no_holdout|drop_holdout)\(",
                       re.MULTILINE)
    exempt = {"ml/model/metrics.py", "scripts/test_holdout.py",
              "ml/holdout.py"}
    missing, seen = [], []
    for path in sorted(list((ROOT / "ml").rglob("*.py"))
                       + list((ROOT / "scripts").rglob("*.py"))):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text()
        if rel in exempt or not scores.search(text):
            continue
        seen.append(rel)
        if not guard.search(text):
            missing.append(rel)
    check("scripts that score a model were found (the scan is not vacuous)",
          len(seen) >= 7, True)
    check("each of them calls assert_no_holdout or drop_holdout in code, "
          "not only in a comment", missing, [])


def case_gates() -> None:
    print("\n6. THE MEASURABILITY GATES MATCH stability.py")
    from ml.model import stability
    check("MIN_POSITIVES == stability.MIN_TEST_POSITIVES",
          MIN_POSITIVES, stability.MIN_TEST_POSITIVES)
    check("MIN_RANKABLE_PAIRS == stability.MIN_RANKABLE_PAIRS",
          MIN_RANKABLE_PAIRS, stability.MIN_RANKABLE_PAIRS)


def case_final_eval(tmp: pathlib.Path) -> None:
    print("\n7. final_eval.py IS SEALED, KEEPS A LEDGER, AND TRAINS ON DEV")
    ledger = tmp / "data" / "holdout_ledger.csv"
    script = "ml/model/final_eval.py"

    code, out = run(script, tmp)
    check("without --unseal it refuses and writes nothing",
          (code != 0, "Sealed" in out, ledger.exists()), (True, True, False))

    code, out = run(script, tmp, "--unseal")
    check("the first --unseal runs", code, 0)
    if code:
        print(out[-2000:])
        return
    first = pd.read_csv(ledger)
    check("and is recorded as the first opening",
          first["reason"].tolist(), ["first opening"])
    check("it reports all three metrics",
          all(s in out for s in ("PR-AUC", "precision@10", "nDCG@20")), True)
    n_dev = len(pd.read_csv(tmp / "data" / "features.csv"))
    n_hold = len(pd.read_csv(tmp / "data" / "holdout.csv"))
    check("the ledger says it trained on every dev row, scored every "
          "holdout row", (int(first["dev_rows"].iloc[0]),
                          int(first["holdout_rows"].iloc[0])),
          (n_dev, n_hold))

    # The number itself, recomputed here from dev rows ALONE. A final_eval
    # that quietly fitted on dev + holdout would record the same row counts
    # and a better PR-AUC; this is what catches it.
    got = probe(tmp, (
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
        "m, _, _ = fit_cv(d, f, 'label_alias', 'lambdarank')\n"
        "h['s'] = score_with(m, h, f)\n"
        "print('EXPECTED', evaluate(h, 's', 'label_alias')['pr_auc'])\n"))
    exp = re.search(r"EXPECTED ([0-9.]+)", got)
    check("its PR-AUC is exactly what training on dev rows alone gives",
          bool(exp) and abs(float(exp.group(1))
                            - float(first["pr_auc"].iloc[0])) < 1e-6, True)
    if not exp:
        print(got[-1500:])

    code, out = run(script, tmp, "--unseal")
    check("a second opening without --again is refused, ledger unchanged",
          (code != 0, len(pd.read_csv(ledger))), (True, 1))

    code, out = run(script, tmp, "--unseal", "--again", "  ")
    check("--again with an empty reason is refused",
          (code != 0, len(pd.read_csv(ledger))), (True, 1))

    code, out = run(script, tmp, "--unseal", "--again",
                    "test: metric code changed")
    check("--again with a reason runs and the reason is recorded",
          (code, pd.read_csv(ledger)["reason"].tolist()[-1]),
          (0, "test: metric code changed"))

    # Slip one TRAINING row into holdout.csv. Scored as if unseen, it would
    # flatter the result by exactly the amount the model memorised it.
    hold_path = tmp / "data" / "holdout.csv"
    good = hold_path.read_text()
    one_dev_row = pd.read_csv(tmp / "data" / "features.csv").head(1)
    pd.concat([pd.read_csv(hold_path), one_dev_row]).to_csv(hold_path,
                                                           index=False)
    code, out = run(script, tmp, "--unseal", "--again", "tamper test")
    check("a holdout.csv carrying a training row is refused, not scored",
          (code != 0, len(pd.read_csv(ledger))), (True, 2))
    hold_path.write_text(good)


def case_serving(tmp: pathlib.Path) -> None:
    print("\n8. train.py QUOTES NO PRE-FREEZE NUMBER; THE SITE KEEPS ITS "
          "NEWEST RELEASES")
    # A stability file from before the freeze: no holdout_from stamp. Its
    # late cuts scored holdout rows, so its range must not reach model_run.
    stab = tmp / "data" / "stability_label_cv.csv"
    sweep = pd.DataFrame({"cut": ["2026-05-01", "2026-06-01"],
                          "skipped": [False, False],
                          "lift_vs_pop": [1.5, 1.7],
                          "beats_pop": [True, True]})
    sweep.to_csv(stab, index=False)
    code, out = run("ml/model/train.py", tmp)
    check("train.py runs on the frozen features.csv", code, 0)
    if code:
        print(out[-2000:])
        return
    notes = pd.read_json(tmp / "artifacts" / "metrics.json",
                         typ="series")["notes"]
    check("a pre-freeze stability file is named stale and not quoted",
          ("predates the holdout freeze" in out, "across" in notes),
          (True, False))
    # Control: the same file WITH the stamp is quoted, so the check above
    # fails if train.py simply stopped quoting stability files at all.
    sweep.assign(holdout_from=str(HOLDOUT_START.date())).to_csv(
        stab, index=False)
    code, out = run("ml/model/train.py", tmp)
    notes = pd.read_json(tmp / "artifacts" / "metrics.json",
                         typ="series")["notes"]
    check("a stamped stability file is quoted as before",
          (code, "across 2 cut dates" in notes), (0, True))
    stab.unlink()

    scored = ("import pandas as pd\n"
              "from ml.db import _text, score_everything\n"
              "keys = set(score_everything())\n"
              "t = {'version_from': str, 'version_to': str}\n"
              "h = pd.read_csv('data/holdout.csv', dtype=t)\n"
              "hk = {(r.package, r.version_to, r.symbol, r.kind, "
              "_text(r, 'sub_target')) for r in h.itertuples(index=False)}\n"
              "print('SCORED', len(keys), len(hk & keys), len(hk))\n")
    got = re.search(r"SCORED (\d+) (\d+) (\d+)", probe(tmp, scored))
    check("db.py scores every holdout row for serving",
          bool(got) and got.group(2) == got.group(3) != "0", True)

    # Two ways that used to go wrong, found in review. An EMPTY holdout.csv
    # (an older dataset) turned every feature to object dtype and LightGBM
    # refused; versions that all look like numbers came back as floats.
    hold_path = tmp / "data" / "holdout.csv"
    good = hold_path.read_text()
    hold = pd.read_csv(hold_path)
    hold.head(0).to_csv(hold_path, index=False)
    got = re.search(r"SCORED (\d+) (\d+) (\d+)", probe(tmp, scored))
    n_dev_keys = probe(tmp, (
        "import pandas as pd\n"
        "from ml.db import _text\n"
        "d = pd.read_csv('data/features.csv', dtype=str)\n"
        "print('DEV', len({(r.package, r.version_to, r.symbol, r.kind, "
        "_text(r, 'sub_target')) for r in d.itertuples(index=False)}))\n"))
    dev_keys = re.search(r"DEV (\d+)", n_dev_keys)
    check("with an empty holdout.csv, db.py still scores every dev row",
          bool(got and dev_keys) and got.group(1) == dev_keys.group(1), True)
    hold.assign(version_from="2.9", version_to="2.10").to_csv(hold_path,
                                                             index=False)
    got = probe(tmp, ("from ml.db import score_everything\n"
                      "v = {k[1] for k in score_everything()}\n"
                      "print('HAS', '2.10' in v, '2.1' in v)\n"))
    check("versions that look like numbers stay text ('2.10', not 2.1)",
          "HAS True False" in got, True)
    hold_path.write_text(good)


def main() -> None:
    labelled = make_labelled()
    case_boundary()
    case_partition(labelled)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-holdout-"))
    try:
        (tmp / "data").mkdir()
        labelled.to_csv(tmp / "data" / "labelled.csv", index=False)
        case_build(tmp, labelled)
        built = all((tmp / "data" / f).exists()
                    for f in ("features.csv", "holdout.csv"))
        if built:
            case_tripwire(tmp, labelled)
        case_structure()
        case_gates()
        if built:
            case_final_eval(tmp)
            case_serving(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDo not quote a number until this passes. A failure here "
              "means some\nexperiment can see the holdout, and the final "
              "evaluation would no longer\nbe a fair exam.")
        sys.exit(1)
    print("All checks passed. Holdout rows reach no experiment, the stale")
    print("pre-freeze file is refused everywhere, and the one sealed path")
    print("to the holdout keeps a record of every opening.")


if __name__ == "__main__":
    main()
