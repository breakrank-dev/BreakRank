"""
Item 15 (F25, NOTES §36): which ranker ships, now that path length alone
matches the model (§34.2). Decided by the rule in §36.1, written before
any of the new rankers was scored on the real data.

    python scripts/ship_verdict.py
    python scripts/ship_verdict.py data/stability_label_alias_cv_tuned_at-20261011.csv

Reads  the sweep `ml/model/stability.py --at ...` wrote at the seven
       reference dates on 11 Oct, kept aside as
       data/stability_label_alias_cv_tuned_at-20261011.csv (the default)
Writes nothing. Nothing is fitted either: the sweep fitted and scored
       every ranker at every date on the same test half.

THE RANKERS, simplest first (stability.CONTEST holds their columns).

  P     path length alone: minus public_depth, nothing fitted (§34.1)
  R     the two reachability features, public_depth and has_export_path,
        chosen and fitted exactly as the shipped model is
  M     the shipped model, 17 features
  P->M  path length, its ties ordered by the model
  R->M  R, its ties ordered by the model

Every one is measured the same way: PR-AUC with ties averaged (F42), and
nDCG@20 and precision@10 averaged over every order of its ties (F44).

ONE TEST. X beats Y when, at the seven dates, X wins on at least one of
§34.1's two measures by §34.1's own rule (path_verdict.across and
within) and Y wins on neither:

  across releases    PR-AUC: X/Y above 1 at 6+ of 7 and its median 1.25x
                     or more (Y: below 1 at 6+, median 0.80x or less)
  within an upgrade  nDCG@20 higher at 6+ of 7

THE RULE (§36.1).

  1. R and M are each set against P. One that does not beat P is out.
  2. Neither beats P: P stands. Only R does: R stands. Only M does: M
     stands, unless R beats M, and then P. Both do: R stands, unless M
     beats R.
  3. If P or R stands, the order inside its ties: the model's (P->M or
     R->M) if its nDCG@20 is higher at 6+ of 7 dates; otherwise no order
     is claimed inside a tie. Judged on nDCG@20 alone, where a release's
     order shows.
  Whatever stands ships, with that order inside its ties.

IT CHECKS ITSELF. No verdict unless the sweep passes every check
path_verdict.py makes (stamped as the shipped model is run, exactly the
seven dates, none skipped, F41's columns), has item 15's columns with no
gap, and was measured on §34.2's data: at every date the same number of
test rows and positives, and path length's three numbers as §34.2 stored
them, exactly. Path length is fitted on nothing, so on the same rows in
the same file they cannot move. The model is refitted, and where its
numbers differ from §34.2's the verdict says so beside its own; they do
not stop it (§36.1).
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ml.model.stability import CONTEST, CONTEST_EXTRA  # noqa: E402
from path_verdict import (AGAIN, DATES, KEPT_SHIP, across,  # noqa: E402
                          refusal, within)

DEFAULT = KEPT_SHIP

# The five rankers by the names the rule uses, in CONTEST's order.
SHORT = dict(zip(["P", "R", "M", "P->M", "R->M"], CONTEST))
COLS = {k: CONTEST[name] for k, name in SHORT.items()}

# §34.2's sweep, 10 Oct: every date's test half, path length's numbers,
# which nothing fitted can move, and the model's. path_ndcg_20 and
# path_p_at_10 are the fixed-order numbers §34.2 was ruled on; the sweep
# still writes them, beside the averaged ones the rule reads.
REFERENCE = {
    "2025-08-07": dict(test_rows=11027, test_pos=307, path=0.1880,
                       path_ndcg_20=0.6037, path_p_at_10=0.2914,
                       pr_auc_ties=0.2298, ndcg_20=0.5611, p_at_10=0.2829),
    "2025-10-06": dict(test_rows=10271, test_pos=296, path=0.1956,
                       path_ndcg_20=0.6037, path_p_at_10=0.2941,
                       pr_auc_ties=0.2571, ndcg_20=0.5796, p_at_10=0.2676),
    "2025-12-03": dict(test_rows=9666, test_pos=279, path=0.2232,
                       path_ndcg_20=0.5855, path_p_at_10=0.3067,
                       pr_auc_ties=0.3054, ndcg_20=0.5817, p_at_10=0.2767),
    "2026-01-18": dict(test_rows=7959, test_pos=259, path=0.2289,
                       path_ndcg_20=0.6028, path_p_at_10=0.3192,
                       pr_auc_ties=0.2033, ndcg_20=0.6094, p_at_10=0.2962),
    "2026-03-02": dict(test_rows=4914, test_pos=178, path=0.2088,
                       path_ndcg_20=0.5758, path_p_at_10=0.2952,
                       pr_auc_ties=0.1698, ndcg_20=0.5352, p_at_10=0.2238),
    "2026-04-02": dict(test_rows=4253, test_pos=155, path=0.2537,
                       path_ndcg_20=0.5490, path_p_at_10=0.2941,
                       pr_auc_ties=0.2721, ndcg_20=0.6748, p_at_10=0.2706),
    "2026-05-04": dict(test_rows=1623, test_pos=140, path=0.2928,
                       path_ndcg_20=0.5720, path_p_at_10=0.3133,
                       pr_auc_ties=0.2446, ndcg_20=0.6691, p_at_10=0.3000),
}
DATA = ("test_rows", "test_pos", "path", "path_ndcg_20", "path_p_at_10")
MODEL = ("pr_auc_ties", "ndcg_20", "p_at_10")


def values(t: pd.DataFrame, ranker: str, measure: int) -> list[float]:
    """One ranker's PR-AUC (measure 0), nDCG@20 (1) or precision@10 (2),
    a value per date in DATES order, as the sweep stores it."""
    return t.loc[DATES, COLS[ranker][measure]].astype(float).tolist()


def beats(t: pd.DataFrame, x: str, y: str) -> dict:
    """§36.1's one test: x wins on at least one of §34.1's measures and y
    on neither. path_verdict's functions call their first argument "model"
    and their second "path"; here they are x and y."""
    a = across(values(t, x, 0), values(t, y, 0))
    w = within(values(t, x, 1), values(t, y, 1))
    wins = a["outcome"] == "model" or w["outcome"] == "model"
    loses = a["outcome"] == "path" or w["outcome"] == "path"
    return {"beats": wins and not loses, "across": a, "within": w}


def decide(t: pd.DataFrame) -> dict:
    """§36.1's rule, step by step. Returns the ranker that ships and each
    comparison made on the way, as (challenger, holder, result)."""
    steps, out = [], []
    for x in ("R", "M"):
        b = beats(t, x, "P")
        steps.append((x, "P", b))
        if b["beats"]:
            out.append(x)
    if not out:
        holder = "P"
    elif out == ["R"]:
        holder = "R"
    elif out == ["M"]:
        # M has beaten P, but not if the simpler R beats M: then M's edge
        # over P is in doubt, and R, which did not beat P, cannot stand.
        b = beats(t, "R", "M")
        steps.append(("R", "M", b))
        holder = "P" if b["beats"] else "M"
    else:
        b = beats(t, "M", "R")
        steps.append(("M", "R", b))
        holder = "M" if b["beats"] else "R"
    ships = holder
    if holder in ("P", "R"):
        broken = f"{holder}->M"
        w = within(values(t, broken, 1), values(t, holder, 1))
        steps.append((broken, holder, {"within": w}))
        if w["outcome"] == "model":
            ships = broken
    return {"ships": ships, "steps": steps}


def contest_refusal(t: pd.DataFrame) -> str | None:
    """Why item 15 cannot be ruled on from this file, or None. Run after
    path_verdict.refusal, which checks the stamps, the dates and F41's
    columns."""
    need = ([c for cols in COLS.values() for c in cols] + list(CONTEST_EXTRA)
            + list(DATA) + list(MODEL))
    missing = [c for c in dict.fromkeys(need) if c not in t]
    if missing:
        return ("This sweep has no " + ", ".join(missing) + " column: it was "
                "written before item 15\n(NOTES §36), or not by a sweep of "
                "the shipped model. " + AGAIN)
    if t[[c for c in dict.fromkeys(need)]].isna().any().any():
        return "The sweep has an empty value at one of the seven dates."
    return None


def other_data(t: pd.DataFrame) -> list[str]:
    """Where the sweep was not measured on §34.2's data: a date's test rows,
    positives or path length's numbers not as §34.2 stored them. Empty
    when it was."""
    bad = []
    for d, ref in REFERENCE.items():
        for c in DATA:
            got = float(t.loc[d, c])
            if abs(got - ref[c]) > 5e-5:
                bad.append(f"{d} {c}: {got:g}, §34.2 had {ref[c]:g}")
    return bad


def model_moved(t: pd.DataFrame) -> list[str]:
    """Where the refitted model's numbers are not §34.2's, as stored. They
    are reported, not refused (§36.1)."""
    moved = []
    for d, ref in REFERENCE.items():
        for c in MODEL:
            got = float(t.loc[d, c])
            if abs(got - ref[c]) > 5e-5:
                moved.append(f"{d} {c} {got:.4f} (§34.2: {ref[c]:.4f})")
    return moved


def describe(x: str, y: str, b: dict) -> str:
    """One comparison as the report reads it."""
    w = b["within"]
    who = {"model": x, "path": y, "none": "neither"}
    nd = (f"    nDCG@20: {x} higher at {w['ahead']}/7, {y} higher at "
          f"{w['behind']}/7; wins: {who[w['outcome']]}\n")
    if "across" not in b:
        return (f"  {x} against {y}: the order inside {y}'s ties\n" + nd
                + f"    {y}'s ties ordered by the model: "
                + ("YES" if w["outcome"] == "model" else "no"))
    a = b["across"]
    return (f"  {x} against {y}\n"
            f"    PR-AUC: {x} ahead at {a['wins']}/7, behind at "
            f"{a['losses']}/7, median {x}/{y} {a['median']:.4f}x; "
            f"wins: {who[a['outcome']]}\n" + nd
            + f"    {x} beats {y}: {'YES' if b['beats'] else 'no'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="NOTES §36.1: which ranker "
                                 "ships.")
    ap.add_argument("sweep", nargs="?", default=str(DEFAULT),
                    help=f"the --at sweep to read (default {DEFAULT})")
    args = ap.parse_args()
    f = pathlib.Path(args.sweep)
    if not f.exists():
        sys.exit(f"{f} not found. " + AGAIN)
    t = pd.read_csv(f, dtype={"cut": str, "commit": str, "test_hash": str})
    why = refusal(t) or contest_refusal(t)
    if why:
        sys.exit(why)
    t = t.set_index("cut")
    bad = other_data(t)
    if bad:
        sys.exit("This sweep was not measured on §34.2's data, so it is not "
                 "the exam item 15 was\nwritten for:\n  " + "\n  ".join(bad)
                 + "\nNo verdict. By §36.1 nothing is ruled on until NOTES "
                 "says why the data moved.")

    print("\nITEM 15 (NOTES §36): WHICH RANKER SHIPS, AT THE SEVEN DATES OF "
          "§23.5")
    print(f"  {f}")
    env = t.loc[DATES[0]]
    print(f"  ran on LightGBM {env['lightgbm']}, scikit-learn "
          f"{env['sklearn']}, pandas {env['pandas']}, numpy {env['numpy']}, "
          f"Python {env['python']},\n  {int(env['cpus'])} CPUs, "
          f"OMP_NUM_THREADS {env['omp_threads']}, commit {env['commit']}")
    print("  §34.2's data: the same test rows and positives at every date, "
          "and path length's\n  numbers exactly as §34.2 stored them. Test "
          "halves: " + ", ".join(t.loc[DATES, "test_hash"].astype(str)))
    moved = model_moved(t)
    if moved:
        print("  the model, refitted: NOT §34.2's at\n    "
              + "\n    ".join(moved)
              + "\n  so §34.1's list keeps §34.2's verdict, and step 1's M "
              "against P is recorded\n  beside it (§36.1)")
    else:
        print("  the model, refitted: its numbers are §34.2's, so step 1's M "
              "against P is §34.2's\n  exam with F44's averaging, and "
              "§34.1's list follows it (§36.1)")
    print()
    for k, name in SHORT.items():
        print(f"  {k:<5} {name}")
    for measure, title in ((0, "PR-AUC, ties averaged (P->M and R->M "
                               "not ruled on)"),
                           (1, "nDCG@20, averaged over tie orders"),
                           (2, "precision@10, the same, not ruled on")):
        print(f"\n  {'':<12}" + "".join(f"{k:>8}" for k in COLS)
              + f"     {title}")
        cols = {k: values(t, k, measure) for k in COLS}
        for i, d in enumerate(DATES):
            print(f"  {d:<12}" + "".join(f"{cols[k][i]:>8.4f}" for k in COLS))
        print(f"  {'median':<12}" + "".join(
            f"{pd.Series(cols[k]).median():>8.4f}" for k in COLS))
    trees = t.loc[DATES, "reach_trees"].astype(int)
    levels = t.loc[DATES, "reach_levels"].astype(int)
    print(f"\n  R: {trees.min()} – {trees.max()} trees, {levels.min()} – "
          f"{levels.max()} distinct scores; settings "
          + ", ".join(sorted(set(t.loc[DATES, "reach_setting"].astype(str)))))

    v = decide(t)
    print("\nTHE RULE, NOTES §36.1, fixed before this run\n")
    for x, y, b in v["steps"]:
        print(describe(x, y, b))
        print()
    print(f"SHIPS: {v['ships']}, {SHORT[v['ships']]}.")
    print("\n  What follows is NOTES §36.1's list for that outcome.")


if __name__ == "__main__":
    main()
