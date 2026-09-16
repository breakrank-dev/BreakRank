"""
Does the result survive more than one arbitrary date?

    python ml/model/stability.py
    python ml/model/stability.py --label label_alias
    python ml/model/stability.py --all-labels

Reads  data/features.csv
Writes data/stability_<label>.csv

WHY THIS IS NOTES ITEM 0 AND NOT A NICE-TO-HAVE.

Every number this project has reported comes from ONE cut date. Everything
older trains, everything newer tests. That single choice decides the test
positive rate, which decides PR-AUC's floor, which decides whether a lift
looks like 1.8x or 3.1x. It also decides the validation slice, which
decides where early stopping halts — measured anywhere between 1 and 120
trees across nine ablation runs on the same data.

That is not a robustness footnote. It produced a WRONG CLAIM. NOTES §5.3
reported "path shape alone reaches 105% of the full model, therefore the
model IS the depth heuristic". Re-running with consistent feature groups
showed every subset beating the full model under that label — dropping
popularity alone doubled PR-AUC. A 15-feature model losing to every subset
of itself is an unstable evaluation, not a hidden heuristic. The 105% was
variance, read as a finding, and it sat in the notebook for days.

So: cut at SEVERAL dates, refit at each, and report the spread. A result
that only holds at one cut is not a result, it is a coincidence with a
decimal point.

WHAT THIS DOES NOT DO. These splits are not independent — they share most
of their training data and overlap heavily in test — so the spread is a
sensitivity range, not a confidence interval, and it must never be written
with a +/- that implies statistics it does not have. Report it as "median,
and the range across N cut dates". That is an honest claim and a useful
one: it says how much the answer moves when the arbitrary choice moves.

THE BASELINES ARE REFITTED AT EVERY CUT TOO. kind_prior is fitted on train
only, and popularity's PR-AUC depends on the test half's composition, so
carrying one cut's baseline across all of them would compare a moving
model against a fixed target and flatter whichever way the data leaned.
Lift is computed WITHIN each split, then summarised.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.model.baselines import add_baseline_scores  # noqa: E402
from ml.model.metrics import evaluate, n_rankable  # noqa: E402
from ml.model.train import (GROUP, fit_cv, fit_model,  # noqa: E402
                            prepare, score_with, split_valid)

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"

# Where to cut. Spread across the middle of the date range: earlier than
# 0.55 leaves too little training data, later than 0.85 leaves a test half
# with almost no positives, and both ends produce numbers that say more
# about the cut than about the model.
CUTS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

MIN_TEST_POSITIVES = 30

# A cut with four rankable version pairs is not a weak measurement of
# precision@10. It is not a measurement of precision@10.
#
# A pair is rankable at 10 only if it has more than 10 changes AND at least
# one positive — otherwise ordering it is either impossible or trivially
# correct. The q=0.85 cut leaves 4 or 5 such pairs, and its lift was the
# WORST CASE for both candidate labels, which is to say the ship decision
# was resting on the least measurable cut in the sweep.
#
# ADDED 14 SEP, AFTER SEEING IT MATTER — and that has to be said out loud,
# because choosing a rule once you can see who it favours is the same
# family of error as picking features on test PR-AUC (§5.4).
#
# Two things defend it. The justification never references which label
# wins: four pairs cannot support a top-10 metric no matter whose model is
# being scored, and MIN_TEST_POSITIVES was already conceding that tiny
# cuts are invalid — it just gated on the wrong quantity. And it is
# checkable against data collected before the rule existed:
#
#     lift_MIN          with q=0.85        without it
#     12 Sep dataset    alias  1.94 win    alias  2.19 win
#     14 Sep dataset    scoped 1.92 win    alias  2.32 win
#
# The ungated rule reversed when 50 rows (0.26% of the data) were added,
# because the cut dates are QUANTILES and every one of them moves when the
# row count does. The gated rule gives the same answer on both datasets.
# A decision rule that flips on a quarter of a percent of the data is not
# measuring the thing it claims to measure.
MIN_RANKABLE_PAIRS = 10


def one_split(df: pd.DataFrame, q: float, label: str, feats: list[str],
              objective: str, stopping: str = "cv") -> dict | None:
    """Train and score at one cut date. None if the split is unusable."""
    when = pd.to_datetime(df["released_at"], errors="coerce")
    cutoff = when.dropna().quantile(q)

    is_test = when > cutoff
    # Undated rows train, never test — the same rule as build.temporal_split.
    test = df[is_test.fillna(False)].sort_values(GROUP)
    full_train = df[~is_test.fillna(False)].sort_values(GROUP)

    pos = int(test[label].sum())
    # Checked BEFORE fitting — an unusable cut should not cost a model.
    rankable = n_rankable(test, label, 10)
    if (pos < MIN_TEST_POSITIVES or rankable < MIN_RANKABLE_PAIRS
            or full_train.empty):
        return {"cut": str(cutoff.date()), "q": q, "test_rows": len(test),
                "test_pos": pos, "rankable10": rankable, "skipped": True}

    train, valid, _ = split_valid(full_train)
    train, valid = train.sort_values(GROUP), valid.sort_values(GROUP)
    if train.empty or valid.empty or train[label].sum() == 0:
        return {"cut": str(cutoff.date()), "q": q, "test_rows": len(test),
                "test_pos": pos, "rankable10": rankable, "skipped": True}

    if stopping == "cv":
        model, trees, _folds = fit_cv(full_train, feats, label, objective)
        fit_on = full_train
    else:
        model = fit_model(train, valid, feats, label, objective)
        trees = getattr(model, "best_iteration_", None) or 600
        fit_on = train

    scored = test.copy()
    scored["model"] = score_with(model, test, feats)
    scored = add_baseline_scores(fit_on, scored, label)

    m = evaluate(scored, "model", label)
    pop = evaluate(scored, "popularity", label)
    sem = evaluate(scored, "semver", label)
    floor = float(test[label].mean())

    return {
        "cut": str(cutoff.date()),
        "q": q,
        "test_rows": len(test),
        "test_pos": pos,
        "floor": round(floor, 4),
        "trees": trees,
        "rankable10": n_rankable(test, label, 10),
        "pr_auc": round(m["pr_auc"], 4),
        "p_at_10": round(m["precision_at_10"], 4),
        "ndcg_20": round(m["ndcg_at_20"], 4),
        "popularity": round(pop["pr_auc"], 4),
        "semver": round(sem["pr_auc"], 4),
        "lift_vs_pop": round(m["pr_auc"] / pop["pr_auc"], 2)
        if pop["pr_auc"] else float("nan"),
        "beats_pop": bool(m["pr_auc"] > pop["pr_auc"]),
        "beats_semver": bool(m["pr_auc"] > sem["pr_auc"]),
        "skipped": False,
    }


def run_label(df: pd.DataFrame, label: str, feats: list[str],
              objective: str, stopping: str = "cv") -> pd.DataFrame:
    rows = [one_split(df, q, label, feats, objective, stopping) for q in CUTS]
    return pd.DataFrame([r for r in rows if r])


def report(t: pd.DataFrame, label: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {label}   —   {len(t)} cut dates")
    print("=" * 74)

    skipped = t[t["skipped"]]
    t = t[~t["skipped"]].copy()
    if skipped.shape[0]:
        for _, r in skipped.iterrows():
            why = []
            if int(r.test_pos) < MIN_TEST_POSITIVES:
                why.append(f"{int(r.test_pos)} test positives "
                           f"(min {MIN_TEST_POSITIVES})")
            rk = r.get("rankable10")
            if pd.notna(rk) and int(rk) < MIN_RANKABLE_PAIRS:
                why.append(f"{int(rk)} rankable pairs "
                           f"(min {MIN_RANKABLE_PAIRS})")
            print(f"  skipped q={r.q:.2f} ({r.cut}): "
                  + ", ".join(why or ["no usable train/valid split"]))
    if t.empty:
        print("  no usable splits.")
        return

    cols = ["cut", "test_rows", "test_pos", "floor", "trees", "rankable10",
            "pr_auc", "popularity", "lift_vs_pop", "p_at_10", "ndcg_20"]
    print()
    print(t[cols].to_string(index=False))

    def band(col: str) -> str:
        v = t[col].astype(float)
        return (f"{v.median():.4f}   range {v.min():.4f} – {v.max():.4f}"
                f"   spread {v.max() - v.min():.4f}")

    print(f"\n  PR-AUC        {band('pr_auc')}")
    print(f"  floor         {band('floor')}")
    print(f"  lift vs pop   {t['lift_vs_pop'].median():.2f}x   "
          f"range {t['lift_vs_pop'].min():.2f}x – {t['lift_vs_pop'].max():.2f}x")
    print(f"  trees         {int(t['trees'].min())} – {int(t['trees'].max())}")

    wins = int(t["beats_pop"].sum())
    gate = int(t["beats_semver"].sum())
    n = len(t)
    print(f"\n  beats popularity at {wins}/{n} cut dates")
    print(f"  beats semver (the kill-date gate) at {gate}/{n}")

    if wins == n:
        print("\n  ** Holds at every cut date. The result is not one lucky")
        print("  ** date, and that is the claim worth making — not the "
              "single\n  ** best number in the column.")
    elif wins == 0:
        print("\n  ** Never beats popularity. Whatever the single-split run "
              "said,\n  ** this does not survive. Report the baseline as the "
              "finding.")
    else:
        print(f"\n  ** Beats popularity at only {wins} of {n} cut dates. That "
              "is a\n  ** coin-flip dressed as a result. Do not quote the "
              "best split;\n  ** quote this ratio.")

    v = t["pr_auc"].astype(float)
    if v.median() and (v.max() - v.min()) > 0.5 * v.median():
        print("\n  ** The spread exceeds half the median. Any comparison "
              "between\n  ** two models closer together than that spread is "
              "unreadable —\n  ** including the label-vs-label comparisons.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refit at several cut dates and report the spread.")
    ap.add_argument("--label", default="label_alias",
                    choices=["label", "label_scoped", "label_alias"])
    ap.add_argument("--all-labels", action="store_true",
                    help="run all three and compare them honestly")
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--stopping", default="cv", choices=["cv", "holdout"],
                    help="cv picks the tree count by folds inside train; "
                         "holdout is the old single-slice rule. Run both to "
                         "see whether the fix actually fixed anything.")
    args = ap.parse_args()

    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found — run ml/features/build.py first.")
    df = prepare(pd.read_csv(FEATURES))
    feats = NUMERIC + BOOLEAN + CATEGORICAL

    labels = (["label", "label_scoped", "label_alias"] if args.all_labels
              else [args.label])

    print(f"\n{len(df):,} rows   {len(CUTS)} cut dates   "
          f"{len(feats)} features   objective {args.objective}   "
          f"stopping {args.stopping}")
    print("Each cut refits the model AND the baselines, so lift is computed")
    print("within a split before anything is summarised.")

    summary = {}
    for label in labels:
        t = run_label(df, label, feats, args.objective, args.stopping)
        report(t, label)
        out = DATA / f"stability_{label}_{args.stopping}.csv"
        t.to_csv(out, index=False)
        print(f"\n  saved -> {out}")
        ok = t[~t["skipped"]]
        if not ok.empty:
            summary[label] = ok

    if len(summary) > 1:
        print("\n" + "=" * 74)
        print("  LABEL vs LABEL, ACROSS ALL CUT DATES")
        print("=" * 74)
        # COMPARE ON THE CUTS ALL LABELS SHARE, NOT ON EACH LABEL'S OWN.
        #
        # A cut is skipped per label — it needs enough positives and enough
        # rankable pairs, and a sparse label runs out first. So the labels
        # end up judged on different exams, and lift_MIN is a WORST CASE:
        # the label that was spared the hardest cut wins by not sitting it.
        #
        # Measured 16 Sep. `label` was skipped at q=0.85 for having 9
        # rankable pairs, so it never faced 2026-08-22 — which is exactly
        # where `label_alias` recorded its worst case, 1.45x. On its own
        # five cuts `label` showed 1.81x and won. On the five cuts all
        # three share, `label_alias` shows 1.92x and wins. Same data, same
        # code, opposite conclusion, and only one of them is a comparison.
        #
        # This is the THIRD time this rule has produced a misleading answer
        # (§11.2 was the first, §13.2 the same confusion in the audit). The
        # pattern each time: a number that looks like a measurement of the
        # models is partly a measurement of which rows each one was given.
        common = None
        for t in summary.values():
            cuts = set(t["cut"])
            common = cuts if common is None else (common & cuts)
        common = common or set()
        if any(len(t) != len(common) for t in summary.values()):
            print(f"\n  Comparing on the {len(common)} cut date(s) all labels")
            print("  share. Per-label tables above use every cut that label")
            print("  could use; a worst case taken over different cuts is not")
            print("  a comparison — see the note in the source.")

        rows = []
        for label, t_all in summary.items():
            t = t_all[t_all["cut"].isin(common)] if common else t_all
            if t.empty:
                continue
            v, l_ = t["pr_auc"].astype(float), t["lift_vs_pop"].astype(float)
            rows.append({
                "label": label,
                "splits": len(t),
                "pr_auc_median": round(v.median(), 4),
                "lift_median": round(l_.median(), 2),
                # THE DECIDING COLUMN. Medians hide failure modes: two
                # labels can sit 0.08x apart on the median while one of
                # them collapses to 0.37x at a cut the other handles at
                # 1.48x. A model that is sometimes worse than the dumb
                # baseline is not "slightly behind on average", it is
                # unshippable — you cannot tell in advance which day you
                # are having.
                "lift_MIN": round(l_.min(), 2),
                "lift_max": round(l_.max(), 2),
                "beats_pop": f"{int(t['beats_pop'].sum())}/{len(t)}",
            })
        comp = pd.DataFrame(rows)
        print()
        print(comp.to_string(index=False))

        print("\nRead the LIFT column, not PR-AUC. PR-AUC's floor is the")
        print("positive rate and the three labels have different ones, so")
        print("the raw scores are not comparable even here.")

        by_median = comp.loc[comp["lift_median"].idxmax(), "label"]
        by_worst = comp.loc[comp["lift_MIN"].idxmax(), "label"]
        gaps = sorted(r["lift_median"] for r in rows)
        close = len(gaps) > 1 and (gaps[-1] - gaps[-2]) < 0.5

        if close:
            print(f"\n** Median lift does NOT separate these: the top two are")
            print("** within 0.5x of each other. Ignore that column.")
        print(f"\n** Decide on lift_MIN. '{by_worst}' has the best worst case "
              f"at\n** {comp.loc[comp['lift_MIN'].idxmax(), 'lift_MIN']:.2f}x, "
              f"and wins popularity at "
              f"{comp.loc[comp['lift_MIN'].idxmax(), 'beats_pop']} cut dates.")
        survivors = comp[comp["lift_MIN"] > 1.0]["label"].tolist()
        if survivors:
            print(f"** Never worse than the baseline at any cut: "
                  f"{', '.join(survivors)}.")
        losers = comp[comp["lift_MIN"] <= 1.0]["label"].tolist()
        if losers:
            print(f"** Sometimes LOSES to popularity: {', '.join(losers)}. "
                  "A model\n** that is worse than the dumb baseline on some "
                  "dates cannot be\n** shipped on the strength of its median.")
        if by_median != by_worst:
            print(f"\n** Note: '{by_median}' has the better median but "
                  f"'{by_worst}' has\n** the better worst case. Prefer the "
                  "worst case.")


if __name__ == "__main__":
    main()
