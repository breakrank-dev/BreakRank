"""
Does scripts/compare_sweeps.py compare two sweeps only when they are the
same exam, and read the before and after right?

    python scripts/test_compare_sweeps.py

No data and no model; a few seconds. Hand-built sweep files in a temp
directory.

Three cases:

  1. The reading: every date's lift before -> after; a date counts as
     moved only by more than 0.25x (§24.1), and which way; the medians,
     the worst case and the counts; and nothing is written.
  2. A cut skipped in both files is left out of both; skipped in one
     only, the two did not measure the same dates and are refused.
  3. Refused: different cut dates (with the --at command that fixes it),
     and sweeps run another way (tuning, relevance, holdout boundary).
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_sweeps.py"
DATES = ["2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
         "2026-03-02", "2026-04-02", "2026-05-04"]

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def sweep(lifts: list[float], dates: list[str] = DATES,
          **stamps) -> pd.DataFrame:
    """A stability file: one row per date, PR-AUC = lift x popularity's
    0.05, so the numbers agree with each other."""
    n = len(lifts)
    t = pd.DataFrame({
        "cut": dates[:n], "q": [0.5] * n, "test_rows": [2000 + i for i in
                                                        range(n)],
        "test_pos": [150] * n, "floor": [0.075] * n, "trees": [30] * n,
        "rankable10": [20] * n,
        "pr_auc": [round(0.05 * x, 4) for x in lifts],
        "p_at_10": [0.25] * n, "ndcg_20": [0.55 + i / 100 for i in range(n)],
        "popularity": [0.05] * n, "lift_vs_pop": lifts,
        "lift_lo": [x / 2.5 for x in lifts],
        "lift_hi": [x * 1.7 for x in lifts],
        "beats_pop": [x > 1 for x in lifts],
        "linear": [0.19] * n,
        "beats_linear": [0.05 * x > 0.19 for x in lifts],
        "skipped": [False] * n,
        "holdout_from": "2026-07-28", "relevance": "binary", "tuning": "cv"})
    return t.assign(**stamps)


def run(tmp: pathlib.Path, before: pd.DataFrame,
        after: pd.DataFrame) -> tuple[int, str]:
    before.to_csv(tmp / "before.csv", index=False)
    after.to_csv(tmp / "after.csv", index=False)
    p = subprocess.run([sys.executable, str(SCRIPT), "before.csv",
                        "after.csv"], cwd=tmp, capture_output=True,
                       text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


BEFORE = [5.80, 5.43, 6.04, 5.20, 3.07, 4.39, 2.20]
# up by 0.40, down by 0.30, and five that move by 0.25 or less, one of
# them by exactly 0.25, which the sweep cannot resolve.
AFTER = [6.20, 5.13, 6.10, 5.45, 3.00, 4.39, 2.10]


def case_reading(tmp: pathlib.Path) -> None:
    print("\n1. THE READING")
    files = set(tmp.iterdir())
    code, out = run(tmp, sweep(BEFORE), sweep(AFTER))
    check("it runs", code, 0)
    if code:
        print(out[-1500:])
        return
    lines = {d: next((ln for ln in out.splitlines()
                      if ln.strip().startswith(d)), "") for d in DATES}
    check("each date's lift, before -> after",
          [bool(re.search(rf"{b:.2f}x -> {a:.2f}x", lines[d]))
           for d, b, a in zip(DATES, BEFORE, AFTER)], [True] * 7)
    marks = [("up" if " up " in lines[d] + " " else
              "down" if " down " in lines[d] + " " else "") for d in DATES]
    check("moved only beyond 0.25x: up at the first date, down at the "
          "second, and 0.25 exactly is not a move",
          marks, ["up", "down", "", "", "", "", ""])
    check("the medians and the worst case",
          bool(re.search(r"median 5\.20x -> 5\.13x, worst 2\.20x -> "
                         r"2\.10x", out)), True)
    check("the counts: 2 of 7 dates moved, 1 up and 1 down",
          "2 of 7 dates (1 up, 1 down)" in out, True)
    check("beats popularity, and the interval above 1.0x, before and after",
          ("beats popularity       7/7 -> 7/7" in out,
           "interval above 1.0x    6/7 -> 6/7" in out), (True, True))
    check("nothing is written", set(tmp.iterdir()), files | {
        tmp / "before.csv", tmp / "after.csv"})
    code, out = run(tmp, sweep(BEFORE), sweep(BEFORE))
    check("the same file twice: no date moved",
          "none of the 7 dates" in out, True)


def case_skips(tmp: pathlib.Path) -> None:
    print("\n2. SKIPPED CUTS")
    b, a = sweep(BEFORE), sweep(AFTER)
    b.loc[6, "skipped"] = a.loc[6, "skipped"] = True
    code, out = run(tmp, b, a)
    check("skipped in both: left out of both, so 6 dates",
          (code, "6 cut dates measured by both" in out), (0, True))
    a2 = sweep(AFTER)
    a2.loc[3, "skipped"] = True
    code, out = run(tmp, sweep(BEFORE), a2)
    check("skipped in one only: refused",
          (code != 0, "measured only before: 2026-01-18" in out),
          (True, True))


def case_refusals(tmp: pathlib.Path) -> None:
    print("\n3. NOT THE SAME EXAM")
    moved = DATES[:6] + ["2026-05-11"]
    code, out = run(tmp, sweep(BEFORE), sweep(AFTER, dates=moved))
    check("different cut dates are refused, with the --at command",
          (code != 0, "Different cut dates" in out,
           "stability.py --at " + ",".join(DATES) in out),
          (True, True, True))
    for col, other in (("tuning", "fixed"), ("relevance", "graded"),
                       ("holdout_from", "2026-08-04")):
        code, out = run(tmp, sweep(BEFORE), sweep(AFTER, **{col: other}))
        check(f"a sweep with another {col} is refused",
              (code != 0, f"{col} is" in out), (True, True))


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-sweeps-"))
    try:
        case_reading(tmp)
        case_skips(tmp)
        case_refusals(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Two sweeps are compared only at the same dates")
    print("and run the same way, and a move is counted only beyond 0.25x.")


if __name__ == "__main__":
    main()
