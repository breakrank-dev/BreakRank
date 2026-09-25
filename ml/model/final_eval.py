"""
The final exam. Run it once, for the final report, and not before.

    python ml/model/final_eval.py --unseal --label label_alias
    python ml/model/final_eval.py --unseal --label label_alias --again "why"

Reads  data/features.csv          every row released BEFORE the holdout
       data/holdout.csv           the frozen holdout (ml/holdout.py)
       data/holdout_manifest.csv  its pair list on the day it froze, to
                                  report how the set has moved since
Writes data/holdout_ledger.csv    one row per opening, appended, never
                                  rewritten

WHAT IT DOES. The pipeline as it stands on the day is trained on every
row released before 2026-08-04, then scored once on the holdout. Same
features and the same fitting code as train.py, and the baselines are
refitted on the same rows, so lift is computed inside one split exactly
as everywhere else.

It trains on ALL the dev rows, the old test split included. That split's
job was to help choose, and the choosing is over by the time this runs.
Leaving a quarter of the history out now would only make the final model
worse than the one that gets shipped.

WHY IT REFUSES BY DEFAULT. A holdout is worth something only if it is
looked at once. Every extra look is a chance to adjust something and look
again, and after a few rounds the holdout is just another test split. So:

  - without --unseal it does nothing;
  - the first --unseal run is written to data/holdout_ledger.csv BEFORE
    any number is printed, so a look cannot happen off the record;
  - a second run needs --again "<reason>", and the reason is written too.

A second look is sometimes right: a bug found in metrics.py, say. It is
never invisible. data/ is not committed, so the ledger is a local record;
the block this script prints at the end goes into NOTES, which is.

KEEP IT IN STEP WITH train.py. It must fit exactly what train.py ships.
When F2 (graded relevance) or F7 (tuning) changes how train.py fits, this
file changes in the same commit.
"""

import argparse
import datetime
import pathlib
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import (FROZEN_ON, GROUP, HOLDOUT_FILE,  # noqa: E402
                        HOLDOUT_START, MIN_POSITIVES, MIN_RANKABLE_PAIRS,
                        assert_no_holdout, fingerprint, holdout_mask, track)
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import evaluate, n_rankable  # noqa: E402
from ml.model.train import fit_cv, prepare, score_with  # noqa: E402

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"
LEDGER = DATA / "holdout_ledger.csv"
ROOT = pathlib.Path(__file__).resolve().parents[2]


def git_commit() -> str:
    """Which code produced the number. '+uncommitted' if anything under
    ml/ or scripts/ differs from that commit or is not in git at all,
    because then nobody can check out the code that made it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--",
                                "ml", "scripts"],
                               cwd=ROOT, capture_output=True, text=True,
                               timeout=10).stdout.strip()
    except Exception:
        return "unknown"
    return (sha or "unknown") + ("+uncommitted" if dirty else "")


def load(label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in (FEATURES, HOLDOUT_FILE):
        if not p.exists():
            sys.exit(f"{p} not found. Run ml/features/build.py first.")
    # Pair keys as text on both sides, or versions like "2.10" could come
    # back as the float 2.1 and the overlap check below would compare
    # nothing.
    as_text = {c: str for c in GROUP}
    dev_raw = pd.read_csv(FEATURES, dtype=as_text)
    hold_raw = pd.read_csv(HOLDOUT_FILE, dtype=as_text)

    assert_no_holdout(dev_raw, "final_eval.py (training rows)")
    # And the mirror image: holdout.csv must hold ONLY holdout rows. A file
    # that somehow gained older rows would mean training rows are being
    # scored as if unseen.
    if hold_raw.empty:
        sys.exit(f"{HOLDOUT_FILE} is empty. Nothing to evaluate.")
    stray = ~holdout_mask(hold_raw)
    if stray.any():
        sys.exit(f"{HOLDOUT_FILE} holds {int(stray.sum()):,} rows released "
                 f"before {HOLDOUT_START.date()}. It is not the frozen "
                 "holdout;\nrebuild it with ml/features/build.py.")
    overlap = (dev_raw[GROUP].drop_duplicates()
               .merge(hold_raw[GROUP].drop_duplicates(), on=GROUP))
    if len(overlap):
        sys.exit(f"{len(overlap)} version pairs are in BOTH files. Rebuild "
                 "with ml/features/build.py.")
    if label not in hold_raw.columns:
        sys.exit(f"no column {label} in {HOLDOUT_FILE}")
    # Checked before anything is fitted or recorded: with no positives
    # there is no number to see, so no look has happened.
    if not hold_raw[label].sum():
        sys.exit(f"the holdout has no positives under {label}. There is "
                 "nothing to measure.")

    # One prepare() over both, exactly as train.py prepares one file, so
    # the categorical columns share a single set of categories.
    both = prepare(pd.concat([dev_raw.assign(_part="dev"),
                              hold_raw.assign(_part="holdout")],
                             ignore_index=True))
    dev = both[both["_part"] == "dev"].drop(columns="_part")
    hold = both[both["_part"] == "holdout"].drop(columns="_part")
    return dev.sort_values(GROUP), hold.sort_values(GROUP)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score the finished pipeline on the frozen holdout, once.")
    ap.add_argument("--unseal", action="store_true",
                    help="required. Without it this script does nothing.")
    ap.add_argument("--again", metavar="REASON", default=None,
                    help="required for any opening after the first; the "
                         "reason is written to the ledger")
    ap.add_argument("--label", default="label_alias",
                    choices=["label", "label_scoped", "label_alias"])
    ap.add_argument("--objective", default="lambdarank",
                    choices=["lambdarank", "binary"])
    args = ap.parse_args()

    if not args.unseal:
        sys.exit("Sealed. This scores the final pipeline on the frozen "
                 "holdout, and it is meant\nto run once, for the final "
                 "report. Every opening is written to\n"
                 f"{LEDGER}. If that is what you are doing now, add --unseal.")
    if args.again is not None and not args.again.strip():
        sys.exit("--again needs a reason, in words. It goes in the ledger.")

    prior = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    if len(prior) and args.again is None:
        cols = [c for c in ("opened_at", "label", "commit", "pr_auc",
                            "reason") if c in prior]
        print(prior[cols].to_string(index=False))
        sys.exit(f"\nThe holdout has already been opened {len(prior)} "
                 "time(s), listed above.\nA further look needs --again "
                 "\"<reason>\", and the reason is recorded.")

    label = args.label
    dev, hold = load(label)
    feats = NUMERIC + BOOLEAN + CATEGORICAL
    commit = git_commit()
    if commit.endswith("+uncommitted"):
        print("** Tracked files differ from the last commit. The ledger will "
              "say so, and\n** nobody will be able to check out the code "
              "that made this number.\n** Commit first unless you have a "
              "reason not to.\n")

    model, trees, folds = fit_cv(dev, feats, label, args.objective)
    scored = hold.copy()
    scored["model"] = score_with(model, scored, feats)
    scored = add_baseline_scores(dev, scored, label)

    names = ["model", "popularity", "kind_prior", "semver", "griffe_all"]
    res = {n: evaluate(scored, n, label) for n in names}
    m, pop, sem = res["model"], res["popularity"], res["semver"]
    floor = float(hold[label].mean())
    pos = int(hold[label].sum())
    r10, r20 = n_rankable(scored, label, 10), n_rankable(scored, label, 20)
    pairs = len(hold[GROUP].drop_duplicates())
    lift = m["pr_auc"] / pop["pr_auc"] if pop["pr_auc"] else float("nan")
    moved_text, moved = track(hold, write=False)

    # THE LEDGER FIRST, THEN THE NUMBERS. If printing crashed after the
    # results were on screen, a look would have happened with no record.
    row = {
        "opened_at": datetime.datetime.now(datetime.timezone.utc)
                     .isoformat(timespec="seconds"),
        "frozen_on": FROZEN_ON,
        "holdout_from": str(HOLDOUT_START.date()),
        "commit": commit,
        "label": label,
        "objective": args.objective,
        "trees": trees,
        "cv_folds": " ".join(map(str, folds)),
        "dev_rows": len(dev),
        "holdout_rows": len(hold),
        "holdout_pairs": pairs,
        "fingerprint": fingerprint(hold),
        "frozen_pairs": moved.get("frozen_pairs", ""),
        "pairs_gone": moved.get("gone", ""),
        "pairs_added": moved.get("added", ""),
        "positives": pos,
        "floor": round(floor, 6),
        "pr_auc": round(m["pr_auc"], 6),
        "popularity": round(pop["pr_auc"], 6),
        "semver": round(sem["pr_auc"], 6),
        "lift_vs_pop": round(lift, 4),
        "precision_at_10": round(m["precision_at_10"], 6),
        "rankable10": r10,
        "ndcg_at_20": round(m["ndcg_at_20"], 6),
        "rankable20": r20,
        "reason": args.again or "first opening",
    }
    pd.DataFrame([row]).to_csv(LEDGER, mode="a", index=False,
                               header=not LEDGER.exists())

    print(f"\nFINAL EVALUATION ON THE FROZEN HOLDOUT   (opening "
          f"{len(prior) + 1}, recorded in {LEDGER})")
    print("=" * 72)
    print(f"  holdout    pairs released on or after {HOLDOUT_START.date()}, "
          f"frozen {FROZEN_ON}")
    print(f"             {len(hold):,} rows   {pairs:,} version pairs   "
          f"{hold['package'].nunique():,} packages   "
          f"fingerprint {row['fingerprint']}")
    print(" " * 13 + moved_text.strip().replace("\n  ", "\n" + " " * 13))
    print(f"  trained on {len(dev):,} rows released before it   label "
          f"{label}   {args.objective}")
    print(f"             CV folds chose {folds} -> {trees} trees   "
          f"code {commit}")
    print()
    print(f"  floor (positive rate)   {floor:.4f}   ({pos} positives)")
    for n in names:
        print(f"  {n:<12} PR-AUC     {res[n]['pr_auc']:.4f}")
    print(f"\n  lift over popularity    {lift:.2f}x")
    print(f"  model vs floor          {m['pr_auc'] / floor:.1f}x")
    print(f"  beats semver (the gate) "
          f"{'yes' if m['pr_auc'] > sem['pr_auc'] else 'NO'}")

    def per_pair(name: str, value: float, n: int) -> str:
        if n < MIN_RANKABLE_PAIRS:
            return (f"  {name:<15} {value:.4f} over {n} upgrades  <- BELOW "
                    f"the {MIN_RANKABLE_PAIRS}-pair gate. Quote it only "
                    "with that n, as an anecdote.")
        return f"  {name:<15} {value:.4f} over {n} upgrades"

    print(per_pair("precision@10", m["precision_at_10"], r10))
    print(per_pair("nDCG@20", m["ndcg_at_20"], r20))
    if pos < MIN_POSITIVES:
        print(f"\n** Only {pos} positives. PR-AUC on this few is fragile; "
              "say so beside it.")

    print("\n" + "-" * 72)
    print("  For NOTES, exactly once:")
    print(f"  Frozen holdout (pairs released on or after "
          f"{HOLDOUT_START.date()}, frozen {FROZEN_ON}, fingerprint "
          f"{row['fingerprint']}),")
    print(f"  opened {row['opened_at']} at commit {commit}: label {label}, "
          f"{len(hold):,} rows / {pairs} pairs, floor {floor:.4f}.")
    if moved:
        print(f"  Against the frozen list of {moved['frozen_pairs']} pairs: "
              f"{moved['gone']} gone, {moved['added']} added.")
    print(f"  PR-AUC {m['pr_auc']:.4f} vs popularity {pop['pr_auc']:.4f} "
          f"({lift:.2f}x) and semver {sem['pr_auc']:.4f};")
    print(f"  precision@10 {m['precision_at_10']:.4f} over {r10} upgrades; "
          f"nDCG@20 {m['ndcg_at_20']:.4f} over {r20}.")


if __name__ == "__main__":
    main()
