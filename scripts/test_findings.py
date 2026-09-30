"""
Item 4 of the fix list (F9, F10): does item4_findings.py count what it
says, and word it by the rules?

    python scripts/test_findings.py

No network and no real data; about ten seconds. Nothing in your data/ or
artifacts/ is read or touched.

WHY THIS FILE EXISTS. Both findings are rates, and a rate can be made to
look sure of itself in two quiet ways: count a symbol once per parameter
row, and draw the interval from changes that are not independent. Each
makes the interval narrower than the data supports, and a narrow
interval is what turns "no detectable difference" into "zero signal".

Five cases:

  1. A change is counted once: a symbol's parameter rows are one change,
     and rows that disagree about it are counted and reported.
  2. The buckets have §19.2's edges: 0 | 1-5 | 6-20 | 21+.
  3. The intervals resample whole upgrades. When positives cluster in
     one release, the interval is much wider than one that resampled
     changes, which is checked here as the control.
  4. The three wording rules, at their edges.
  5. The script, end to end: it runs on a built features.csv, reports
     both findings under both labels, reads the gain shares and says
     when they come from a model that does not ship, and refuses a
     features.csv that still holds holdout rows.
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

from ml.holdout import HOLDOUT_START  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


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


F = load("item4_findings")


def rows(spec: list[tuple]) -> pd.DataFrame:
    """(package, symbol, n rows, label, deprecated, prior) per change, all
    in upgrades 1.0 -> 1.1 of their package."""
    out = []
    for pkg, sym, n, lab, dep, prior in spec:
        for i in range(n):
            out.append({"package": pkg, "version_from": "1.0",
                        "version_to": "1.1", "symbol": sym,
                        "label_alias": lab, "label": lab,
                        "was_deprecated_before": dep,
                        "prior_breaks_in_module": prior})
    return pd.DataFrame(out)


def case_changes() -> None:
    print("\n1. A CHANGE IS COUNTED ONCE")
    df = rows([("a", "a.f", 49, 1, 1, 0), ("a", "a.g", 1, 0, 0, 3),
               ("b", "a.f", 2, 0, 0, 0)])
    ch, mixed = F.changes(df, "label_alias")
    check("49 parameter rows of one symbol are one change; the same "
          "symbol in another package is another",
          (len(ch), sorted(ch["positive"])), (3, [0, 0, 1]))
    check("clean rows: nothing reported as disagreeing", mixed, 0)
    df.loc[0, "prior_breaks_in_module"] = 7
    ch, mixed = F.changes(df, "label_alias")
    check("rows that disagree are counted, and take their largest value",
          (mixed, int(ch.loc[ch["symbol"].eq("a.f")
                             & ch["package"].eq("a"), "prior"].iloc[0])),
          (1, 7))


def case_buckets() -> None:
    print("\n2. THE BUCKETS")
    got = F.bucket(pd.Series([0, 1, 5, 6, 20, 21, 400])).tolist()
    check("0 | 1-5 | 6-20 | 21+, edges included where §19.2 put them",
          got, ["0", "1-5", "1-5", "6-20", "6-20", "21+", "21+"])


def case_intervals() -> None:
    print("\n3. THE INTERVALS RESAMPLE WHOLE UPGRADES")
    # 40 deprecated changes, every one used, all in one release; one
    # unused deprecated change in each of 40 others. Half the deprecated
    # changes are used, and one release decides it.
    spec = [("big", f"big.f{i}", 1, 1, 1, 0) for i in range(40)]
    for p in range(40):
        spec.append((f"p{p}", f"p{p}.old", 1, 0, 1, 0))
        spec += [(f"p{p}", f"p{p}.g{i}", 1, int(i == 0), 0, 0)
                 for i in range(10)]
    ch, _ = F.changes(rows(spec), "label_alias")
    dep = ch["deprecated"].map({1: "deprecated", 0: "not deprecated"})
    n, k, r, _, _ = F.resample(ch, dep, ["deprecated", "not deprecated"])
    check("the point rates are the plain counts: 40/80 and 40/400",
          (k[0] / n[0], k[1] / n[1]), (0.5, 0.1))
    lo, hi = F.interval(r[:, 0])
    # The control: the same resampling, of changes instead of upgrades.
    rng = np.random.default_rng(0)
    y = ch.loc[ch["deprecated"] == 1, "positive"].to_numpy()
    naive = [rng.choice(y, len(y)).mean() for _ in range(2000)]
    nlo, nhi = F.interval(np.array(naive))
    check(f"one release deciding it shows: {hi - lo:.2f} wide, against "
          f"{nhi - nlo:.2f} resampling changes", (hi - lo) > 2 * (nhi - nlo),
          True)


def case_rules() -> None:
    print("\n4. THE WORDING RULES")
    check("an interval containing 0 is 'no detectable difference', with "
          "the range it cannot rule out",
          F.deprecation_verdict(-0.001, -0.02, 0.019),
          "no detectable difference. The data are consistent with anything "
          "from -2.00 to +1.90 points.")
    check("one that excludes 0 says which way",
          (F.deprecation_verdict(-0.03, -0.05, -0.01).startswith(
              "deprecated changes are used less often"),
           F.deprecation_verdict(0.03, 0.01, 0.05).startswith(
              "deprecated changes are used more often")), (True, True))
    check("the middle peak needs 1-5 or 6-20 clearly above 0",
          (F.peak_verdict({"1-5": -0.01, "6-20": 0.002})[0],
           F.peak_verdict({"1-5": -0.01, "6-20": 0.0})[0]), (True, False))
    check("the collapse needs 21+ clearly below the rest",
          (F.collapse_verdict(-0.03, -0.04, -0.001)[0],
           F.collapse_verdict(-0.03, -0.04, 0.0)[0]), (True, False))


def run(tmp: pathlib.Path):
    p = subprocess.run([sys.executable, str(ROOT / "scripts" /
                                            "item4_findings.py")],
                       cwd=tmp, capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def case_script(tmp: pathlib.Path) -> None:
    print("\n5. THE SCRIPT, END TO END")
    (tmp / "data").mkdir()
    load("test_holdout").make_labelled().to_csv(tmp / "data" /
                                                "labelled.csv", index=False)
    p = subprocess.run([sys.executable, str(ROOT / "ml/features/build.py")],
                       cwd=tmp, capture_output=True, text=True, timeout=600)
    check("build.py exits cleanly", p.returncode, 0)
    if p.returncode:
        return
    code, out = run(tmp)
    check("it runs, and reports both findings under both labels",
          (code, out.count("DEPRECATED BEFORE IT WAS REMOVED"),
           out.count("EARLIER BREAKS IN THE SAME MODULE"),
           "label_alias   (ships)" in out), (0, 2, 2, True))
    saved = tmp / "data" / "item4_findings.csv"
    check("and saves every number it printed",
          saved.exists() and len(pd.read_csv(saved)) == 2 * (3 + 5), True)
    check("with no gain table, it says so rather than inventing one",
          "importance.csv not found" in out, True)

    art = tmp / "artifacts"
    art.mkdir()
    pd.DataFrame({"feature": ["public_depth", "prior_breaks_in_module",
                              "was_deprecated_before"],
                  "gain": [60.0, 30.0, 10.0]}).to_csv(art / "importance.csv",
                                                      index=False)
    (art / "metrics.json").write_text('{"version": "lambdarank-label"}')
    code, out = run(tmp)
    check("gain shares are read and ranked, and a model that is not the "
          "shipped one is named as such",
          (code, "prior_breaks_in_module    30.0%   rank 2 of 3" in out,
           "not the shipped model" in out), (0, True, True))

    feats = tmp / "data" / "features.csv"
    stale = pd.read_csv(feats)
    stale.loc[0, "released_at"] = HOLDOUT_START.strftime("%Y-%m-%d")
    stale.to_csv(feats, index=False)
    code, out = run(tmp)
    check("a features.csv holding a holdout row is refused",
          (code != 0, "HOLDOUT ROWS" in out), (True, True))


def main() -> None:
    case_changes()
    case_buckets()
    case_intervals()
    case_rules()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-findings-"))
    try:
        case_script(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Each change counts once, the intervals")
    print("resample whole upgrades, and the findings say only what they show.")


if __name__ == "__main__":
    main()
