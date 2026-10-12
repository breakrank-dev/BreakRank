"""
NOTES §36.2: path length alone ships. Does the code that ships it do what
§36.2's list says, and refuse what it should?

    python scripts/test_path_ships.py

No network, no database, no real data; a few minutes. Hand-made frames,
the labelled fixture test_ship.py builds, and the real build.py,
train.py, final_eval.py and verdict scripts on it in a temp directory.

Eight cases:

  1. One convention: evaluate(), precision@10, nDCG@20 and intervals()
     average over every order of tied scores unless told otherwise, and
     the old way is still there to ask for. tie_averaged() adds path
     length's lift over popularity, on the same resamples intervals()
     draws.
  2. ml/model/ship.py: path length's score, a missing depth scored as the
     deepest of the frame or of its own release; the name of its
     model_run row; the seven dates path_verdict.py reads.
  3. The sweep: path length's lift over popularity with its interval on
     every row, the convention stamp, and the report, which reads both
     and still reads a sweep from before them as before.
  4. train.py: path length's model_run row, its numbers the test half's
     own, its range across both sweeps with a shared date counted once;
     the model's row on the same convention; and neither row quoting a
     sweep from before the convention, or another label's row taking the
     served name.
  5. The loader: path length's scores for every row, features and
     holdout, a missing depth the deepest of its release, under
     path_metrics.json's row; --scores loads them unless told --ranker
     model; db_model_check.py asks after the row that ships.
  6. stories.py: a list's hits counted by their average over the order
     of its ties, a tied change placed at the middle of its tie, and the
     tie printed as a span.
  7. What refuses: compare_sweeps.py across the two conventions;
     path_verdict.py and ship_verdict.py on a sweep that has the stamp.
  8. final_eval.py: path length first, its numbers in the ledger with the
     convention; a ledger with other columns stops it before any number.
"""

import contextlib
import importlib.util
import io
import json
import os
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

from ml import db  # noqa: E402
from ml.holdout import HOLDOUT_START, assert_no_holdout  # noqa: E402
from ml.model import ship, stability  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import (CONVENTION, evaluate,  # noqa: E402
                              intervals, ndcg_at_k, precision_at_k,
                              tie_averaged)
from ml.model.ship import path_scores  # noqa: E402
from ml.model.train import prepare, stability_name  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []
LABEL = "label_alias"
GROUP = ["package", "version_from", "version_to"]
STAMP = str(HOLDOUT_START.date())


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


@contextlib.contextmanager
def inside(path: pathlib.Path):
    """Run with `path` as the working directory, as the scripts expect:
    their data/ and artifacts/ are relative."""
    here = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(here)


def run(script: str, cwd: pathlib.Path, *args: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(ROOT / script), *args],
                       cwd=cwd, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


# ------------------------------------------------------ 1. one convention

def tied() -> pd.DataFrame:
    """Forty upgrades whose model, path and popularity scores all tie."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(40):
        n = int(rng.integers(12, 30))
        for i in range(n):
            rows.append({"package": f"p{u % 6}", "version_from": f"{u}",
                         "version_to": f"{u}.1",
                         "y": int(i == 0 or rng.random() < 0.2),
                         "model": float(rng.integers(0, 4)),
                         "popularity": float(-(u % 6)),
                         "path": float(-rng.integers(1, 4))})
    return pd.DataFrame(rows)


def case_convention() -> None:
    print("\n1. ONE CONVENTION: TIES AVERAGED, UNLESS TOLD OTHERWISE")
    df = tied()
    check("the convention's name, as sweeps and notes carry it",
          CONVENTION, "ties-averaged")
    check("evaluate() averages over tie orders by default, and the old way "
          "is still there and differs here",
          (evaluate(df, "model", "y")
           == evaluate(df, "model", "y", ties=True, expected=True),
           evaluate(df, "model", "y")
           != evaluate(df, "model", "y", ties=False, expected=False)),
          (True, True))
    check("so do precision@10 and nDCG@20 on their own",
          (precision_at_k(df, "model", "y", 10)
           == precision_at_k(df, "model", "y", 10, expected=True),
           ndcg_at_k(df, "model", "y", 20)
           == ndcg_at_k(df, "model", "y", 20, expected=True),
           precision_at_k(df, "model", "y", 10)
           != precision_at_k(df, "model", "y", 10, expected=False)),
          (True, True, True))
    a = intervals(df, "y", "model", "popularity", draws=200)
    b = intervals(df, "y", "model", "popularity", draws=200, ties=True,
                  expected=True)
    c = intervals(df, "y", "model", "popularity", draws=200, ties=False,
                  expected=False)
    check("intervals() too: every PR-AUC and both within-upgrade numbers, "
          "and the old way differs on each",
          (a == b, a["pr_auc"] != c["pr_auc"],
           a["precision_at_10"] != c["precision_at_10"],
           a["ndcg_at_20"] != c["ndcg_at_20"]), (True, True, True, True))
    t = tie_averaged(df, "y", draws=200)
    pp = intervals(df, "y", "path", "popularity", draws=200,
                   per_upgrade=False)
    mp = intervals(df, "y", "model", "popularity", draws=200,
                   per_upgrade=False)
    mt = intervals(df, "y", "model", "path", draws=200, per_upgrade=False)
    check("tie_averaged(): path length's lift over popularity and its "
          "band, the band intervals() gives on the same resamples, and the "
          "model's two as before",
          (abs(t["path_pop"] - t["path"]["pr_auc"] / t["popularity"])
           < 1e-12, t["ci_path_pop"] == pp["lift"], t["ci_pop"] == mp["lift"],
           t["ci_path"] == mt["lift"], t["path"] == evaluate(df, "path", "y")),
          (True, True, True, True, True))


# --------------------------------------------------------------- 2. ship.py

def case_ship() -> None:
    print("\n2. ship.py: THE SCORE, ITS NAME, ITS DATES")
    f = pd.DataFrame({"package": ["a", "a", "a", "b", "b", "c"],
                      "version_from": ["1", "1", "1", "2", "2", "3"],
                      "version_to": ["1.1", "1.1", "1.1", "2.1", "2.1", "3.1"],
                      "public_depth": [1, 2, np.nan, 3, np.nan, np.nan]},
                     index=[40, 10, 30, 20, 50, 60])
    frame = path_scores(f)
    release = path_scores(f, within=GROUP)
    check("minus public_depth, a missing depth the frame's deepest",
          frame.to_dict(), {40: -1.0, 10: -2.0, 30: -3.0, 20: -3.0,
                            50: -3.0, 60: -3.0})
    check("or, for the site, its own release's deepest; a release with no "
          "depth at all scores 0; aligned to the rows' own index",
          release.to_dict(), {40: -1.0, 10: -2.0, 30: -2.0, 20: -3.0,
                              50: -3.0, 60: 0.0})
    check("the served name is path-length for the label that ships, and "
          "carries any other label's name",
          (ship.path_version(LABEL), ship.path_version("label")),
          ("path-length", "path-length-label"))
    check("what ships, and the seven dates are path_verdict.py's",
          (ship.SHIPPED_RANKER, ship.RANKERS,
           list(ship.REFERENCE_DATES) == load("path_verdict").DATES,
           str(ship.PATH_METRICS)),
          ("path", ("path", "model"), True, "artifacts/path_metrics.json"))
    test = load("test_ship").dev().head(400).copy()
    test["public_depth"] = pd.to_numeric(test["public_depth"]).astype(float)
    test.iloc[0, test.columns.get_loc("public_depth")] = np.nan
    scored = add_baseline_scores(test, test, LABEL)
    check("baselines.py's path column is ship.path_scores, exactly, a "
          "missing depth scored as the frame's deepest",
          (scored["path"].equals(path_scores(test)),
           scored["path"].iloc[0] == -float(test["public_depth"].max())),
          (True, True))


# ------------------------------------------------------------- 3. the sweep

def sweep_table(stamped: bool) -> pd.DataFrame:
    t = pd.DataFrame({
        "cut": ["2026-01-01", "2026-02-01"], "q": [0.6, 0.7],
        "test_rows": 100, "test_pos": 40, "floor": 0.1, "trees": 20,
        "rankable10": 12, "pr_auc": 0.3, "popularity": 0.2,
        "lift_vs_pop": 1.5, "lift_lo": [1.1, 1.2], "lift_hi": [1.9, 2.1],
        "p_at_10": 0.3, "ndcg_20": 0.6,
        "beats_pop": True, "beats_semver": True, "skipped": False,
        "pr_auc_ties": 0.3, "popularity_ties": 0.2,
        "lift_vs_pop_ties": 1.5, "lift_lo_ties": 1.1, "lift_hi_ties": 1.9,
        "beats_pop_ties": True, "path": [0.25, 0.32], "path_p_at_10": 0.2,
        "path_ndcg_20": [0.55, 0.58], "lift_vs_path": [1.2, 0.9375],
        "beats_path": [True, False], "beats_path_ndcg": [True, True],
        "path_lift_vs_pop": [1.5, 2.5], "path_lift_lo": [0.9, 1.6],
        "path_lift_hi": [2.4, 3.9], "path_beats_pop": [True, True]})
    return t.assign(convention=CONVENTION) if stamped else t


def report_of(t: pd.DataFrame) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stability.report(t, LABEL)
    return buf.getvalue()


def case_sweep(df: pd.DataFrame) -> None:
    print("\n3. THE SWEEP: PATH LENGTH'S LIFT, THE STAMP, THE REPORT")
    cut = "2026-05-15"
    row = stability.one_split(df, cut, LABEL, ship_feats(), "lambdarank",
                              "cv", "binary", "fixed")
    _, is_test = stability.cut(df, cut)
    test = df[is_test].sort_values(stability.GROUP)
    full = df[~is_test].sort_values(stability.GROUP)
    sc = add_baseline_scores(full, test, LABEL)
    pth = evaluate(sc, "path", LABEL)["pr_auc"]
    pop = evaluate(sc, "popularity", LABEL)["pr_auc"]
    band = intervals(sc, LABEL, "path", "popularity",
                     per_upgrade=False)["lift"]
    check("each row carries path length's lift over popularity, its paired "
          "interval and whether it is ahead, worked out here",
          (row["path_lift_vs_pop"], row["path_lift_lo"], row["path_lift_hi"],
           row["path_beats_pop"]),
          (round(pth / pop, 2), round(band[0], 2), round(band[1], 2),
           bool(pth > pop)))
    check("and its other numbers take the convention: the model's PR-AUC "
          "is its tie-averaged one, nDCG@20 the averaged one",
          (row["pr_auc"] == row["pr_auc_ties"], row["lift_lo"]
           == row["lift_lo_ties"]), (True, True))
    t = stability.run_label(df, LABEL, ship_feats(), "lambdarank", "cv",
                            [cut], tuning="fixed")
    check("every row says the convention it was measured on, and item 15's "
          "fits are off unless asked for",
          (list(t["convention"]), "reach" in t), ([CONVENTION], False))
    level = stability.path_over_popularity(
        {"path": {"pr_auc": 0.2}, "popularity": 0.2,
         "ci_path_pop": (0.91, 1.09)})
    none = stability.path_over_popularity(
        {"path": {"pr_auc": 0.2}, "popularity": 0.0,
         "ci_path_pop": (float("nan"), float("nan"))})
    check("path length level with popularity is not ahead, and a popularity "
          "of zero gives no ratio rather than an error",
          (level["path_lift_vs_pop"], level["path_beats_pop"],
           level["path_lift_lo"], level["path_lift_hi"],
           bool(np.isnan(none["path_lift_vs_pop"]))),
          (1.0, False, 0.91, 1.09, True))

    out = report_of(sweep_table(stamped=True))
    check("the report reads path length's lift on a stamped sweep: the "
          "count, the count clear of its interval, the median and range",
          ("path length alone, the ranker that ships (NOTES §36.2): beats "
           "popularity at 2/2,\n  by more than its interval at 1/2; lift vs "
           "pop median 2.00x, range 1.50x – 2.50x" in out),
          True)
    head = next(ln for ln in out.splitlines() if ln.strip().startswith("cut"))
    check("and shows its columns, not the F42 ones, which repeat the plain "
          "ones on this convention",
          ("path_lift_vs_pop" in head, "path_lift_95" in head,
           "pr_auc_ties" in head, "with ties averaged (F42)" in out),
          (True, True, False, False))
    old = report_of(sweep_table(stamped=False).drop(
        columns=["path_lift_vs_pop", "path_lift_lo", "path_lift_hi",
                 "path_beats_pop"]))
    head = next(ln for ln in old.splitlines() if ln.strip().startswith("cut"))
    check("a sweep from before the convention is read as it always was",
          ("with ties averaged (F42): lift vs pop median 1.50x" in old,
           "pr_auc_ties" in head, "ranker that ships" in old),
          (True, True, False))


def ship_feats() -> list[str]:
    from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC
    return NUMERIC + BOOLEAN + CATEGORICAL


# --------------------------------------------------------------- 4. train.py

QUANTILE = ["2025-08-07", "2025-10-10", "2025-12-05", "2026-01-21",
            "2026-03-05", "2026-04-05", "2026-05-06"]


def fake_sweep(dates: list[str], lifts: list[float],
               stamped: bool = True) -> pd.DataFrame:
    """A sweep as train.py reads one: stamped as --tuning fixed runs it,
    the model's lift and path length's at each date."""
    t = pd.DataFrame({"cut": dates, "skipped": False,
                      "lift_vs_pop": [x + 0.5 for x in lifts],
                      "beats_pop": True,
                      "lift_lo": [x - 0.1 for x in lifts],
                      "path_lift_vs_pop": lifts,
                      "path_beats_pop": [x > 1 for x in lifts],
                      "path_lift_lo": [x - 0.6 for x in lifts],
                      "holdout_from": STAMP, "relevance": "binary",
                      "tuning": "fixed"})
    return t.assign(convention=CONVENTION) if stamped else t


def range_text(t: pd.DataFrame) -> str:
    v = t["path_lift_vs_pop"].astype(float)
    return (f" | across {len(t)} cut dates: beats popularity "
            f"{int(t['path_beats_pop'].sum())}/{len(t)}, lift vs popularity "
            f"median {v.median():.2f}x min {v.min():.2f}x max {v.max():.2f}x"
            f", lift's 95% interval above 1.0x at "
            f"{int((t['path_lift_lo'] > 1).sum())}/{len(t)}")


def case_train(tmp: pathlib.Path) -> bool:
    print("\n4. train.py: PATH LENGTH'S model_run ROW")
    data = tmp / "data"
    q_lifts = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
    at_lifts = [2.0, 1.0, 1.1, 2.5, 4.0, 0.9, 3.5]
    quant = fake_sweep(QUANTILE, q_lifts)
    at = fake_sweep(list(ship.REFERENCE_DATES), at_lifts)
    q_file = data / stability_name(LABEL, "cv", "lambdarank", "binary",
                                   tuning="fixed")
    at_file = data / stability_name(LABEL, "cv", "lambdarank", "binary",
                                    at=True, tuning="fixed")
    quant.to_csv(q_file, index=False)
    at.to_csv(at_file, index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    check("train.py runs", code, 0)
    if code:
        print(out[-2500:])
        return False
    row = json.loads((tmp / "artifacts" / "path_metrics.json").read_text())
    feats = prepare(pd.read_csv(data / "features.csv"))
    test = feats[feats["split"] == "test"].sort_values(GROUP)
    sc = test.assign(path=path_scores(test))
    want = {k: round(v, 6) for k, v in evaluate(sc, "path", LABEL).items()}
    check("path_metrics.json is path length's row: its name, and PR-AUC, "
          "precision@10 and nDCG@20 on the test half, by the convention",
          (row["version"], {k: row[k] for k in want}, row["positive_rate"]),
          ("path-length", want, round(float(test[LABEL].mean()), 6)))
    both = pd.concat([quant, at]).drop_duplicates("cut", keep="first")
    check("its notes say the convention, and its range runs across both "
          "sweeps' dates, the shared one counted once: 13, not 14",
          ("convention=ties-averaged" in row["notes"],
           row["notes"].endswith(range_text(both)), len(both)),
          (True, True, 13))
    sc["popularity"] = -pd.to_numeric(test["package_rank"]).fillna(
        pd.to_numeric(test["package_rank"]).max())
    band = intervals(sc, LABEL, "path", "popularity")
    fmt = {"pr_auc_95": ("pr_auc", 4), "lift_95": ("lift", 2),
           "precision_at_10_95": ("precision_at_10", 4),
           "ndcg_at_20_95": ("ndcg_at_20", 4)}
    said = {k: re.search(k + r"=([0-9.]+)-([0-9.]+)", row["notes"])
            for k in fmt}
    check("and they carry path length's own intervals, paired with "
          "popularity's on the test half's upgrades, as intervals() gives "
          "them, and the model beside it",
          ({k: (m.group(1), m.group(2)) if m else None
            for k, m in said.items()},
           all(k in row["notes"] for k in ("model=lambdarank-label_alias:",
                                           "model_over_path="))),
          ({k: (f"{band[name][0]:.{d}f}", f"{band[name][1]:.{d}f}")
            for k, (name, d) in fmt.items()}, True))
    mrow = json.loads((tmp / "artifacts" / "metrics.json").read_text())
    mq = quant.assign(path_lift_vs_pop=quant["lift_vs_pop"],
                      path_beats_pop=quant["beats_pop"],
                      path_lift_lo=quant["lift_lo"])
    check("the model's row takes the convention too, quotes the quantile "
          "sweep alone as before, and drops the F42 notes that now repeat "
          "its own",
          ("convention=ties-averaged" in mrow["notes"],
           range_text(mq) in mrow["notes"], "pr_auc_ties=" in mrow["notes"]),
          (True, True, False))
    check("it prints path length's block, its lift inside its interval",
          ("path length alone, the ranker that ships (NOTES §36.2)" in out
           and lift_inside(out)), True)

    quant.drop(columns="convention").to_csv(q_file, index=False)
    at.drop(columns="convention").to_csv(at_file, index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    row = json.loads((tmp / "artifacts" / "path_metrics.json").read_text())
    mrow = json.loads((tmp / "artifacts" / "metrics.json").read_text())
    check("sweeps from before the convention are quoted in neither row, "
          "and it says why, for each file",
          (code, "across" in row["notes"], "across" in mrow["notes"],
           out.count("measured before 11 Oct")), (0, False, False, 2))
    quant.to_csv(q_file, index=False)
    fake_sweep(["2026-04-01", "2026-05-15"], [2.0, 2.2]).to_csv(at_file,
                                                                index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    row = json.loads((tmp / "artifacts" / "path_metrics.json").read_text())
    check("an --at sweep cut at other dates is not quoted for path length, "
          "and it says so; the quantile sweep's seven dates are",
          (code, "not the seven reference dates" in out,
           row["notes"].endswith(range_text(quant))), (0, True, True))
    q_file.unlink()
    at.to_csv(at_file, index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    row = json.loads((tmp / "artifacts" / "path_metrics.json").read_text())
    mrow = json.loads((tmp / "artifacts" / "metrics.json").read_text())
    check("with the reference-date sweep alone, path length's row quotes "
          "its seven dates and the model's row quotes nothing",
          (code, row["notes"].endswith(range_text(at)),
           "across" in mrow["notes"]), (0, True, False))
    quant.to_csv(q_file, index=False)
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed", "--label",
                    "label")
    row = json.loads((tmp / "artifacts" / "path_metrics.json").read_text())
    check("another label's run writes a row of its own name, never the "
          "served one", (code, row["version"]), (0, "path-length-label"))
    code, out = run("ml/model/train.py", tmp, "--tuning", "fixed")
    return code == 0


def lift_inside(out: str) -> bool:
    block = out.split("the ranker that ships (NOTES §36.2), on the same")[-1]
    m = re.search(r"lift over popularity\s+([0-9.]+)x\s+([0-9.]+)x to "
                  r"([0-9.]+)x", block)
    return bool(m and float(m.group(2)) <= float(m.group(1))
                <= float(m.group(3)))


# --------------------------------------------------------------- 5. loader

def case_loader(tmp: pathlib.Path) -> None:
    print("\n5. THE LOADER, AND WHICH ROW THE SITE SHOULD SERVE")
    data = tmp / "data"
    with inside(tmp):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = db.score_everything("path")
    both = pd.concat([pd.read_csv(data / "features.csv",
                                  dtype={"version_from": str,
                                         "version_to": str}),
                      pd.read_csv(data / "holdout.csv",
                                  dtype={"version_from": str,
                                         "version_to": str})],
                     ignore_index=True)
    want = dict(zip([(r.package, r.version_to, r.symbol, r.kind,
                      db._text(r, "sub_target"))
                     for r in both.itertuples(index=False)],
                    path_scores(both, within=GROUP)))
    check("every row of features.csv and holdout.csv, scored minus its "
          "depth", (len(got), got == want, len(both) > 0),
          (len(want), True, True))
    check("and it says how many had no depth",
          f"{len(both):,} rows scored, 0 with no depth" in out.getvalue(),
          True)

    gap = tmp.parent / "gap"
    shutil.copytree(tmp, gap)
    f = pd.read_csv(gap / "data" / "features.csv",
                    dtype={"version_from": str, "version_to": str})
    # A release whose deepest change is shallower than the frame's, so
    # the two rules give different scores, and a row in it that is not
    # its deepest, so blanking it leaves the release's deepest as it was.
    depth = pd.to_numeric(f["public_depth"])
    by_rel = depth.groupby([f[c] for c in GROUP]).transform("max")
    shallow = f.index[(by_rel < depth.max()) & (depth < by_rel)]
    big = shallow[0]
    rel = f.loc[big, GROUP].tolist()
    same = (f[GROUP] == rel).all(axis=1)
    deepest = float(depth[same].max())
    check("the fixture has such a release", deepest < float(depth.max()),
          True)
    f.loc[big, "public_depth"] = np.nan
    f.to_csv(gap / "data" / "features.csv", index=False)
    r = f.loc[big]
    with inside(gap):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = db.score_everything("path")
    key = (r["package"], r["version_to"], r["symbol"], r["kind"],
           db._text(r, "sub_target"))
    check("a change with no depth scores as the deepest of its own release, "
          "and is counted", (got[key], "1 with no depth" in out.getvalue()),
          (-deepest, True))
    (gap / "artifacts" / "path_metrics.json").unlink()
    with inside(gap):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                db.score_everything("path")
            said = None
        except SystemExit as e:
            said = str(e.code)
    check("with no path_metrics.json it stops, naming the file",
          bool(said and "path_metrics.json not found" in said), True)
    shutil.rmtree(gap, ignore_errors=True)

    L = load("test_db_loader")
    chg, rel_ = L.changes(), L.releases()
    caps = {"sub_target": True, "inherited_by": True, "positive_rate": True,
            "analysis_status": True, "statuses": set(db.DB_STATUSES)}
    usage = pd.DataFrame({"symbol": ["alpha.f1"], "user_count": [4]})
    scores = {(r.package, r.version_to, r.symbol, r.kind, ""): -1.0
              for r in chg.itertuples()}
    row_file = tmp.parent / "row.json"
    row_file.write_text(json.dumps({"version": "path-length-test",
                                    "pr_auc": 0.2, "precision_at_10": 0.3,
                                    "ndcg_at_20": 0.5,
                                    "positive_rate": 0.07, "notes": "x"}))

    class Conn(L.ScoringConn):
        runs: list = []

        def execute(self, stmt, params=None):
            if "INSERT INTO model_run" in str(stmt):
                self.runs.append(dict(params))
            return super().execute(stmt, params)

    conn = Conn(0)
    with contextlib.redirect_stdout(io.StringIO()):
        db.write_all(conn, chg, usage, None, caps, scores, rel_,
                     metrics=row_file)
    check("write_all puts the scores under the row it is given, deleting "
          "only that version's", ([r["version"] for r in conn.runs],
                                  conn.deleted),
          (["path-length-test"], ["path-length-test"]))

    calls = {}

    def spy_score(ranker="model"):
        calls["ranker"] = ranker
        return {}

    def spy_write(*a, metrics=None, **k):
        calls["row"] = metrics

    class Engine:
        @contextlib.contextmanager
        def begin(self):
            yield None

    saved = {n: getattr(db, n) for n in ("score_everything", "write_all",
                                         "connect", "check_schema",
                                         "load_frames", "load_releases")}
    argv = sys.argv
    try:
        db.score_everything, db.write_all = spy_score, spy_write
        db.connect, db.check_schema = (lambda: Engine()), (lambda c, a: {})
        db.load_frames = lambda: (chg, usage, None)
        db.load_releases = lambda: None
        seen = []
        for args in (["--scores"], ["--scores", "--ranker", "model"]):
            calls.clear()
            sys.argv = ["db.py", *args]
            with contextlib.redirect_stdout(io.StringIO()):
                db.main()
            seen.append((calls.get("ranker"), str(calls.get("row"))))
    finally:
        for n, v in saved.items():
            setattr(db, n, v)
        sys.argv = argv
    check("--scores loads path length's scores under path_metrics.json; "
          "--ranker model the model's under metrics.json",
          seen, [("path", "artifacts/path_metrics.json"),
                 ("model", "artifacts/metrics.json")])

    D = load("db_model_check")
    model_name = json.loads((tmp / "artifacts" / "metrics.json")
                            .read_text())["version"]
    rows = [("path-length", "2026-10-12 18:00:00", 0.2650, 21452),
            (model_name, "2026-10-05 10:00:00", 0.2890, 21452)]

    class Res:
        def __init__(self, r):
            self.r = r

        def fetchall(self):
            return self.r

        def scalar(self):
            return self.r[0][0]

    class Check:
        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "FROM model_run" in sql:
                return Res(rows)
            if "count(*) FROM breakage" in sql:
                return Res([(27900,)])
            if "date(computed_at)" in sql:
                return Res([("2026-10-12", 21452)])
            return Res([])

    class CheckEngine:
        @contextlib.contextmanager
        def connect(self):
            yield Check()

    outs = []
    saved = D.connect
    try:
        D.connect = lambda: CheckEngine()
        for args in ([], ["--ranker", "model"]):
            sys.argv = ["db_model_check.py", *args]
            buf = io.StringIO()
            with inside(tmp), contextlib.redirect_stdout(buf):
                D.main()
            outs.append(buf.getvalue())
    finally:
        D.connect = saved
        sys.argv = argv
    check("db_model_check.py asks after the row that ships: path length's, "
          "the one a restarted API serves here",
          ("path-length" in outs[0].split("<- artifacts/path_metrics.json")[0]
           .splitlines()[-1],
           "A restarted API serves path-length, the one in "
           "artifacts/path_metrics.json." in outs[0]), (True, True))
    check("and with --ranker model, the fallback's, which it would not "
          "serve and is not meant to; no advice to pin it",
          (f"A restarted API would serve path-length, not {model_name}, the "
           "fallback" in outs[1], "not meant to be served" in outs[1],
           "MODEL_VERSION=" in outs[1]), (True, True, False))


# --------------------------------------------------------------- 6. stories

def case_stories() -> None:
    print("\n6. stories.py: A TIE COUNTED BY ITS AVERAGE")
    S = load("stories")
    g = pd.DataFrame({"package": "r", "version_from": "1", "version_to": "2",
                      "symbol": ["r.b", "r.a", "r.e", "r.c", "r.d", "r.f"],
                      "sub_target": "", "score": [3, 3, 2, 2, 2, 1.0],
                      LABEL: 0, "alias_user_count": 0})
    r = S.ranked(g)
    check("the site's order, a tie by name, and each tie's span and middle",
          (list(r["symbol"]), list(r["first"]), list(r["last"]),
           list(r["place"])),
          (["r.a", "r.b", "r.c", "r.d", "r.e", "r.f"], [1, 1, 3, 3, 3, 6],
           [2, 2, 5, 5, 5, 6], [1.5, 1.5, 4.0, 4.0, 4.0, 6.0]))

    n = 20
    score = np.repeat([3.0, 2.0, 1.0], [8, 6, 6])
    used = np.zeros(n, int)
    users = np.zeros(n, int)
    for i, u in ((0, 5), (9, 3), (12, 9), (16, 2)):
        used[i], users[i] = 1, u
    rel = pd.DataFrame({"package": "s", "version_from": "1",
                        "version_to": "2",
                        "symbol": [f"s.x{i:02d}" for i in range(n)],
                        "sub_target": "", "kind": "OBJECT_REMOVED",
                        "score": score, LABEL: used,
                        "alias_user_count": users})
    t = S.tally(rel, 20).iloc[0]
    check("hits@10 over a tie that straddles the 10th place: 8 places at "
          "1/8, then 2 at 2/6; the users the same way: 5, then 2 x 12/6",
          (round(t["hits10"], 9), round(t["found_users"], 9),
           t["expected10"], round(t["above"], 9)),
          (round(1 + 2 * 2 / 6, 9), 9.0, 2.0, round(1 + 2 * 2 / 6 - 2, 9)))
    bury = rel.assign(score=np.repeat([3.0, 2.0], [8, 12]),
                      **{LABEL: 0, "alias_user_count": 0})
    bury.loc[0, [LABEL, "alias_user_count"]] = [1, 2]
    bury.loc[8, [LABEL, "alias_user_count"]] = [1, 80]
    bury.loc[19, [LABEL, "alias_user_count"]] = [1, 50]
    f = S.pick_failure(bury, S.tally(bury, 20))
    check("the failure is the most-used change whose tie's middle is in the "
          "bottom half (places 9-20, middle 14.5), though the list prints it "
          "9th; not the one printed last", (f["symbol"], int(f["rank"]),
                                            f["place"]), ("s.x08", 9, 14.5))
    g2 = g.assign(public_depth=[2, 1, 3, 2, 2, 3])
    check("stories.py scores by path length unless told otherwise, as the "
          "loader scores the site",
          list(S.score(g2)), list(path_scores(g2, within=GROUP)))
    flat = pd.concat([
        rel.assign(package="t", score=1.0, **{LABEL: 0, "alias_user_count": 0}),
        rel.assign(package="u", score=np.repeat([2.0, 1.0], [10, 10]),
                   **{LABEL: 0, "alias_user_count": 0})], ignore_index=True)
    flat.loc[3, [LABEL, "alias_user_count"]] = [1, 30]
    flat.loc[20 + 17, [LABEL, "alias_user_count"]] = [1, 5]
    f = S.pick_failure(flat, S.tally(flat, 20))
    check("a change tied with its whole list (place 10.5 of 20) was not "
          "ranked, so it is not the failure, however used; the buried one "
          "in the other release is", (f["package"], f["symbol"], f["place"]),
          ("u", "s.x17", 15.5))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        S.show_list(bury, mark="s.x08")
    check("and the list prints the tie as a span, and says what a span is",
          ("9-20" in buf.getvalue(), "a span is a tie" in buf.getvalue()),
          (True, True))


# -------------------------------------------------------------- 7. refusals

def case_refusals(tmp: pathlib.Path) -> None:
    print("\n7. WHAT REFUSES")
    C = load("compare_sweeps")
    old = fake_sweep(QUANTILE, [2.0] * 7, stamped=False)
    new = fake_sweep(QUANTILE, [2.0] * 7)
    said = C.refusal(old, new)
    check("compare_sweeps.py will not set a sweep from before the convention "
          "beside one after it, and will set two after it side by side",
          (bool(said and "convention is (none) before and ties-averaged "
                "after" in said), C.refusal(new, new)), (True, None))
    T = load("test_ship")
    good = T.sweep()
    for script in ("scripts/path_verdict.py", "scripts/ship_verdict.py"):
        good.assign(convention=CONVENTION).to_csv(tmp / "stamped.csv",
                                                  index=False)
        code, out = run(script, tmp, str(tmp / "stamped.csv"))
        check(f"{script.split('/')[1]} refuses a stamped sweep and names the "
              "two it runs on",
              (code != 0, "not applied again" in out,
               "stability_label_alias_cv_tuned_at-20261010.csv" in out,
               "stability_label_alias_cv_tuned_at-20261011.csv" in out),
              (True, True, True, True))
    good.to_csv(tmp / "kept.csv", index=False)
    code, out = run("scripts/ship_verdict.py", tmp, str(tmp / "kept.csv"))
    check("and still rules on a sweep from before it, as on 11 Oct",
          (code, "SHIPS: P, path length alone." in out), (0, True))


# ------------------------------------------------------------ 8. final_eval

def case_final(tmp: pathlib.Path) -> None:
    print("\n8. final_eval.py: PATH LENGTH FIRST, AND IN THE LEDGER")
    code, out = run("ml/model/final_eval.py", tmp, "--unseal", "--tuning",
                    "fixed")
    check("final_eval.py runs", code, 0)
    if code:
        print(out[-2500:])
        return
    ledger = pd.read_csv(tmp / "data" / "holdout_ledger.csv")
    hold = pd.read_csv(tmp / "data" / "holdout.csv")
    sc = hold.assign(path=path_scores(hold),
                     popularity=-hold["package_rank"].fillna(
                         hold["package_rank"].max()))
    pth, pop = evaluate(sc, "path", LABEL), evaluate(sc, "popularity", LABEL)
    r = ledger.iloc[-1]
    check("the ledger row says the convention and the ranker, and carries "
          "path length's numbers on the holdout",
          (r["convention"], r["ranker"], r["path_pr_auc"],
           r["path_lift_vs_pop"], r["path_precision_at_10"],
           r["path_ndcg_at_20"]),
          (CONVENTION, "path", round(pth["pr_auc"], 6),
           round(pth["pr_auc"] / pop["pr_auc"], 4),
           round(pth["precision_at_10"], 6), round(pth["ndcg_at_20"], 6)))
    first = out.find("PATH LENGTH ALONE, THE RANKER THAT SHIPS")
    then = out.find("THE MODEL, WHICH WAS TRIED")
    check("path length is printed first, the model after it, and the NOTES "
          "block leads with it",
          (0 <= first < then, "Path length alone, the ranker\n  that ships"
           in out), (True, True))
    code, out = run("ml/model/final_eval.py", tmp, "--unseal", "--tuning",
                    "fixed", "--again", "a second look, for the test")
    again = pd.read_csv(tmp / "data" / "holdout_ledger.csv")
    check("a second opening appends to a ledger this code wrote",
          (code, len(again), list(again.columns) == list(ledger.columns),
           again["reason"].iloc[-1]), (0, 2, True, "a second look, for the "
                                                   "test"))
    old = ledger.drop(columns=["convention", "ranker", "path_pr_auc",
                               "path_lift_vs_pop", "path_precision_at_10",
                               "path_ndcg_at_20"])
    for name, frame in (("a ledger with other columns", old),
                        ("a header-only ledger with other columns",
                         old.head(0))):
        frame.to_csv(tmp / "data" / "holdout_ledger.csv", index=False)
        before = (tmp / "data" / "holdout_ledger.csv").read_bytes()
        code, out = run("ml/model/final_eval.py", tmp, "--unseal",
                        "--tuning", "fixed", "--again",
                        "a ledger written before 11 Oct")
        check(f"{name} stops it: nothing written, no number shown",
              (code != 0, "has other columns" in out,
               "PATH LENGTH ALONE" in out,
               (tmp / "data" / "holdout_ledger.csv").read_bytes() == before),
              (True, True, False, True))


def main() -> None:
    case_convention()
    case_ship()
    df = load("test_ship").dev()
    # The frame the sweep is run on here is dev only, as the real one is.
    assert_no_holdout(df, "test_path_ships.py")
    case_sweep(df)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-path-ships-"))
    try:
        work = tmp / "work"
        (work / "data").mkdir(parents=True)
        load("test_ship").labelled().to_csv(work / "data" / "labelled.csv",
                                            index=False)
        code, out = run("ml/features/build.py", work)
        check("build.py exits cleanly", code, 0)
        if code:
            print(out[-1500:])
        elif case_train(work):
            case_loader(work)
            case_stories()
            case_refusals(work)
            case_final(work)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Every number takes one convention, and path "
          "length's\nscores, row and stories are made as NOTES §36.2 says.")


if __name__ == "__main__":
    main()
