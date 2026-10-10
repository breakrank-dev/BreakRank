"""
F41 (NOTES §34): does the model beat ranking by path length alone?
Decided by the rule in NOTES §34.1, written before this was first run.

    python scripts/path_verdict.py
    python scripts/path_verdict.py data/stability_label_alias_cv_tuned_at.csv

Reads  the sweep at the seven reference dates that the retrain's
       `ml/model/stability.py --at ...` wrote; by default the shipped
       model's, data/stability_label_alias_cv_tuned_at.csv
Writes nothing. Nothing is fitted either: at every date the sweep already
       scored the model and path length on the same test half.

THE BASELINE (ml/model/baselines.py). path is minus public_depth, the
dots in the shortest public name a user can write for a symbol: shortest
import path first, nothing fitted. Most of its scores tie. Its PR-AUC, and
the model's it is set against (pr_auc_ties), are taken with ties averaged
(F42, NOTES §35): sklearn's convention scores a run of tied scores as one
threshold, which on synthetic test halves put path 8-16% low and would
have favoured the model. precision@10 and nDCG@20 break ties by
metrics.py's fixed random order, as for every baseline.

THE RULE (NOTES §34.1). Two questions, two verdicts, at the seven dates.

  across releases    PR-AUC, ties averaged. At each date r = model / path,
                     the two as stored, to four places, and r to four.
                     the model wins:  r > 1 at 6+ of 7, median r >= 1.25
                     path wins:       r < 1 at 6+ of 7, median r <= 0.80
                     otherwise no difference the rule can see
  within an upgrade  nDCG@20 over the upgrades rankable at 20.
                     the model wins:  its nDCG@20 higher at 6+ of 7
                     path wins:       path's higher at 6+ of 7
                     otherwise no difference the rule can see

Values are read as the sweep stores them, to four places: a tie there is
higher for neither, and a median that reads exactly 1.25x or 0.80x
counts. precision@10 is printed beside the second verdict, not ruled on.

IT CHECKS ITSELF. No verdict unless every row is stamped as the shipped
model is run (label_alias, lambdarank, tuning cv, binary relevance,
holdout from 2026-07-28), the file holds exactly the seven dates of §23.5,
none skipped, and the path and tie-averaged columns are there: a sweep
written before F41 has none.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.holdout import HOLDOUT_START  # noqa: E402
from ml.model.train import (SHIPPED_RELEVANCE,  # noqa: E402
                            SHIPPED_TUNING, stability_name)

# The cut dates of §23.5, where every fix since item 2 has been measured.
DATES = ["2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
         "2026-03-02", "2026-04-02", "2026-05-04"]
LABEL = "label_alias"
OBJECTIVE = "lambdarank"
MARGIN = 1.25   # the median PR-AUC ratio the model needs; 1/MARGIN, path
NEED = 6        # dates of the 7 a side must win, on either measure
COLUMNS = ("pr_auc_ties", "path", "ndcg_20", "path_ndcg_20", "p_at_10",
           "path_p_at_10", "floor")
DEFAULT = pathlib.Path("data") / stability_name(
    LABEL, "cv", relevance=SHIPPED_RELEVANCE, at=True,
    tuning=SHIPPED_TUNING)


def across(model: list[float], path: list[float]) -> dict:
    """§34.1's first verdict, on PR-AUC, one value per date on each side
    in the same order: "model", "path" or "none", with the numbers it
    rests on."""
    if len(model) != len(path):
        raise ValueError("one PR-AUC per date on each side")
    # To four places, as stability.py stores lift_vs_path and prints its
    # median, so the report and the verdict read the same number.
    ratios = [round(m / p, 4) for m, p in zip(model, path)]
    wins = sum(r > 1 for r in ratios)
    losses = sum(r < 1 for r in ratios)
    median = float(pd.Series(ratios).median())
    # Four-place values, so a median that reads exactly 1.25x or 0.80x
    # counts, and floating point does not get a vote.
    if wins >= NEED and median >= MARGIN - 1e-9:
        outcome = "model"
    elif losses >= NEED and median <= 1 / MARGIN + 1e-9:
        outcome = "path"
    else:
        outcome = "none"
    return {"outcome": outcome, "n": len(ratios), "wins": wins,
            "losses": losses, "median": median, "ratios": ratios}


def within(model: list[float], path: list[float]) -> dict:
    """§34.1's second verdict, on nDCG@20: "model", "path" or "none"."""
    if len(model) != len(path):
        raise ValueError("one nDCG@20 per date on each side")
    ahead = sum(m > p for m, p in zip(model, path))
    behind = sum(m < p for m, p in zip(model, path))
    outcome = ("model" if ahead >= NEED else
               "path" if behind >= NEED else "none")
    diffs = [m - p for m, p in zip(model, path)]
    return {"outcome": outcome, "n": len(diffs), "ahead": ahead,
            "behind": behind, "median": float(pd.Series(diffs).median())}


def refusal(t: pd.DataFrame) -> str | None:
    """Why this file cannot be ruled on, or None."""
    again = ("Run the sweep again at the seven dates:\n  python "
             "ml/model/stability.py --at " + ",".join(DATES))
    if "cut" not in t or "skipped" not in t:
        return "This is not a sweep stability.py wrote. " + again
    want = {"label": LABEL, "objective": OBJECTIVE,
            "tuning": SHIPPED_TUNING, "relevance": SHIPPED_RELEVANCE,
            "holdout_from": str(HOLDOUT_START.date())}
    for col, value in want.items():
        got = sorted(t[col].astype(str).unique()) if col in t else []
        if got != [value]:
            return (f"This sweep was not run as the shipped model is: {col} "
                    f"is {', '.join(got) or 'missing'}, not {value}.\n"
                    "(A sweep written before F41 has no label or objective "
                    "stamp.) " + again)
    cuts = sorted(t["cut"].astype(str))
    if cuts != sorted(DATES):
        return ("This sweep was not cut at the seven dates of §23.5.\n"
                f"  its dates: {', '.join(cuts)}\n" + again)
    skipped = t.loc[t["skipped"].astype(str).str.lower().isin(
        {"true", "1"}), "cut"].astype(str).tolist()
    if skipped:
        return (f"The sweep skipped {', '.join(skipped)}. The rule was "
                "written for all seven dates.")
    missing = [c for c in COLUMNS if c not in t]
    if missing:
        return ("This sweep has no " + ", ".join(missing) + " column: it "
                "was written before F41\n(NOTES §34). " + again)
    if t[list(COLUMNS)].isna().any().any():
        return "The sweep has an empty value at one of the seven dates."
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="NOTES §34.1: the model "
                                 "against path length alone.")
    ap.add_argument("sweep", nargs="?", default=str(DEFAULT),
                    help=f"the --at sweep to read (default {DEFAULT})")
    args = ap.parse_args()
    f = pathlib.Path(args.sweep)
    if not f.exists():
        sys.exit(f"{f} not found. Run: python ml/model/stability.py --at "
                 + ",".join(DATES))
    t = pd.read_csv(f, dtype={"cut": str})
    why = refusal(t)
    if why:
        sys.exit(why)
    t = t.set_index("cut").loc[DATES]
    col = {c: t[c].astype(float).tolist() for c in COLUMNS}
    col["pr_auc"] = col["pr_auc_ties"]
    # The table prints the ratios the rule reads, to four places, so a date
    # it counts as lost never reads 1.00x.
    a = across(col["pr_auc"], col["path"])
    w = within(col["ndcg_20"], col["path_ndcg_20"])

    print("\nF41: THE MODEL AGAINST PATH LENGTH ALONE, AT THE SEVEN DATES OF "
          "§23.5")
    print(f"  {f}: {LABEL}, {OBJECTIVE}, tuning {SHIPPED_TUNING}, "
          f"{SHIPPED_RELEVANCE} relevance,\n  holdout from "
          f"{HOLDOUT_START.date()}")
    print("  path = shortest import path first (minus public_depth), "
          "nothing fitted;\n  PR-AUC with ties averaged for both (F42)\n")
    print(f"  {'':<12}{'PR-AUC':^34}{'nDCG@20':^18}{'precision@10':^18}")
    print(f"  {'cut':<12}{'floor':>8}{'model':>8}{'path':>8}{'ratio':>10}"
          f"{'model':>9}{'path':>9}{'model':>9}{'path':>9}")
    for i, d in enumerate(DATES):
        print(f"  {d:<12}{col['floor'][i]:>8.4f}{col['pr_auc'][i]:>8.4f}"
              f"{col['path'][i]:>8.4f}{a['ratios'][i]:>9.4f}x"
              f"{col['ndcg_20'][i]:>9.4f}{col['path_ndcg_20'][i]:>9.4f}"
              f"{col['p_at_10'][i]:>9.4f}{col['path_p_at_10'][i]:>9.4f}")

    p10 = sum(m > p for m, p in zip(col["p_at_10"], col["path_p_at_10"]))
    print(f"  {'median':<12}{'':>8}{pd.Series(col['pr_auc']).median():>8.4f}"
          f"{pd.Series(col['path']).median():>8.4f}{a['median']:>9.4f}x"
          f"{pd.Series(col['ndcg_20']).median():>9.4f}"
          f"{pd.Series(col['path_ndcg_20']).median():>9.4f}"
          f"{pd.Series(col['p_at_10']).median():>9.4f}"
          f"{pd.Series(col['path_p_at_10']).median():>9.4f}")

    print("\nTHE RULE, NOTES §34.1, fixed before this run\n")
    print(f"  1. Across releases, PR-AUC: the model ahead at {a['wins']}/7, "
          f"behind at {a['losses']}/7,\n     median model/path "
          f"{a['median']:.4f}x")
    print(f"     the model wins: ahead at {NEED}+ and median >= "
          f"{MARGIN:.2f}x\n     path wins: behind at {NEED}+ and median "
          f"<= {1 / MARGIN:.2f}x")
    print("     VERDICT: " + {
        "model": "the model beats path length across releases.",
        "path": "path length alone beats the model across releases.",
        "none": "no difference the rule can see across releases."}
        [a["outcome"]])
    print(f"\n  2. Within an upgrade, nDCG@20: the model higher at "
          f"{w['ahead']}/7, path higher at {w['behind']}/7,\n     median "
          f"difference {w['median']:+.4f}")
    print(f"     the model wins: higher at {NEED}+\n     path wins: path "
          f"higher at {NEED}+")
    print("     VERDICT: " + {
        "model": "the model orders an upgrade's changes better than path "
                 "length.",
        "path": "path length alone orders an upgrade's changes better.",
        "none": "no difference the rule can see within an upgrade."}
        [w["outcome"]])
    print(f"     (precision@10, not ruled on: the model higher at {p10}/7)")

    print("\n  What the report says follows NOTES §34.1's list for each "
          "verdict.")


if __name__ == "__main__":
    main()
