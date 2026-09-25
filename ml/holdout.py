"""
The frozen holdout: the releases no experiment is allowed to look at.

    from ml.holdout import assert_no_holdout, split_off

WHAT THIS IS, IN PLAIN WORDS.

Every number in NOTES so far was measured on a test split that was ALSO
used to decide things: which features stay, which label ships, how the
tree count is chosen. Each decision was taken by looking at test scores,
so the test split slowly stopped being a fair exam. It became the exam the
model was coached for, and the headline is optimistic by an amount nobody
can measure (NOTES F13).

A holdout is a slice of the data sealed in an envelope BEFORE the next
round of decisions. Nothing is tuned, compared or chosen on it. At the
very end the finished pipeline is scored on it exactly once, and that is
the number the report can defend: nothing that produced it saw these rows.

THE RULE, fixed on 25 Sep 2026 before any fix it will judge (fix list
21.8, item 1):

    a version pair is in the holdout if its version_to was released
    ON OR AFTER 2026-08-04.

A date and not a random 20%, for the same reason the ordinary split is
temporal (build.py, rule 2): the product predicts for releases that have
not happened yet, so the fair exam is the newest releases. 2026-08-04 is
the date F13 named when it said "freeze the last cut" of the §20 sweep.

Pairs and not rows. Every row of a pair carries version_to's date, so on
clean data the two rules agree. If a pair ever arrives with rows that
disagree, the whole pair goes to the holdout: half an upgrade in training
and half in the holdout would let the model study the other half of the
exam.

WHAT IT PROTECTS, AND WHAT IT CANNOT.

  Protected: every decision from 25 Sep onward. F1, F5, F2, F7, F3 and the
  rest of the fix list are judged on dev rows only.

  Not protected: decisions taken BEFORE 25 Sep. The 18 features, the CV
  stopping rule and the label shortlist were chosen while these rows sat
  inside ordinary test halves; §20's cuts reached to the end of the data.
  So the report says "untouched by every decision since the freeze", not
  "never seen". The first is true; the second is not.

HOW IT IS ENFORCED. Three layers, because any one of them can be forgotten.

  1. build.py writes holdout rows to data/holdout.csv and never to
     data/features.csv. Every evaluation script reads features.csv, so the
     rows are simply not in the file.
  2. Every evaluation script calls assert_no_holdout() straight after
     loading. That catches what layer 1 cannot: a features.csv built
     BEFORE the freeze, still sitting in data/ with the holdout inside it.
  3. scripts/test_holdout.py checks both, and scans ml/ and scripts/ for
     any file that imports the metrics or the fitting functions without
     making the call. A text scan has limits: a notebook, or a script that
     scores with LightGBM directly, is invisible to it. So the rule still
     has to be known, not just enforced.

ml/model/final_eval.py is the only code that scores holdout.csv against
its labels, and it keeps a ledger, so "report it once" is a record rather
than a promise. ml/db.py also reads the file, for serving only (below).

TWO THINGS THAT ARE EASY TO GET WRONG.

  Serving is not evaluating. ml/db.py still scores holdout rows, so the
  site keeps its newest releases. Those scores sit in the database next to
  the usage table, and joining the two to compute a metric before the
  final report is exactly the look this file exists to prevent.

  Label statistics count too, a little. When a decision rests on a
  positive rate ("PARAMETER_MOVED is 0.32% positive, so collapse it", F16),
  compute it with drop_holdout() first. Counting the holdout's positives
  to know whether it can be measured is fine; build.py does it. Scoring a
  model on it is not.

KNOWN CROSSINGS. Three places where information from after the boundary
still reaches dev rows, found in review on 25 Sep and left for the fix
that owns them rather than patched here:

  package_churn   counts a package's rows over the whole file, holdout
                  included. F5 (item 2) replaces it with a past-only count.
  label_scoped    leaf_owners (labels.py) counts symbols over the whole
                  file, so a holdout symbol can flip a dev row's scoped
                  label. Not the strict label, not label_alias.
  prior_breaks_in_module
                  accumulates in VERSION order at ingest. A backport
                  published after a newer major (1.9.5 after 2.0.0) sits
                  earlier in that order than its date says, so its breaks
                  are counted into later dev pairs. build.py counts pairs
                  like this on every run; the fix is an ingest change.

Not to be confused with `train.py --stopping holdout`, an older use of the
same word: there it means the single validation slice inside train that
early stopping used to watch. Nothing to do with this file.
"""

import hashlib
import pathlib
import sys

import pandas as pd

HOLDOUT_START = pd.Timestamp("2026-08-04")
FROZEN_ON = "2026-09-25"
HOLDOUT_FILE = pathlib.Path("data") / "holdout.csv"
MANIFEST = pathlib.Path("data") / "holdout_manifest.csv"
GROUP = ["package", "version_from", "version_to"]

# The gates stability.py applies to any test half (NOTES §11.2). Copied, not
# imported, so build.py need not load LightGBM just to print a report.
# scripts/test_holdout.py fails if the two copies ever disagree.
MIN_POSITIVES = 30
MIN_RANKABLE_PAIRS = 10


def released(df: pd.DataFrame) -> pd.Series:
    """released_at as timestamps, compared in UTC. Missing dates are NaT.

    run_ingest writes plain 10-character dates, and the fast parse below
    handles those. It infers ONE format from the first value, though, and
    turns anything else into NaT, which would send a real post-boundary
    row to dev. So any value the fast parse drops is parsed again on its
    own ("mixed"), and timezones are converted rather than allowed to
    crash a comparison.
    """
    raw = df["released_at"]
    when = pd.to_datetime(raw, errors="coerce", utc=True)
    missed = when.isna() & raw.notna()
    if missed.any():
        when[missed] = pd.to_datetime(raw[missed], errors="coerce",
                                      utc=True, format="mixed")
    return when.dt.tz_localize(None)


def holdout_mask(df: pd.DataFrame) -> pd.Series:
    """True for every row whose version PAIR belongs to the holdout.

    An undated pair stays out of it: nothing shows it came out on or after
    the boundary. It trains, as undated rows always have in build.py, and
    build.py says how many there are (§21.7 counted none).
    """
    when = released(df)
    keys = [df[c] for c in GROUP]
    newest = when.groupby(keys, dropna=False, sort=False).transform("max")
    return newest.ge(HOLDOUT_START).fillna(False).astype(bool)


def split_off(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dev, holdout). dev is every row an experiment may use."""
    m = holdout_mask(df)
    return df[~m].copy(), df[m].copy()


def drop_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Only the dev rows. For code whose input legitimately holds the
    holdout (changes.csv, labelled.csv) and which must not use it."""
    return df[~holdout_mask(df)].copy()


def assert_no_holdout(df: pd.DataFrame, who: str) -> None:
    """The tripwire. Stop the script if any holdout row reached it."""
    m = holdout_mask(df)
    if not m.any():
        return
    rows = int(m.sum())
    pairs = len(df.loc[m, GROUP].drop_duplicates())
    sys.exit(
        f"\n** {who}: HOLDOUT ROWS IN THE DATA. Refusing to run. **\n"
        f"** {rows:,} rows ({pairs:,} version pairs) were released on or "
        f"after {HOLDOUT_START.date()},\n"
        "** and those belong to the frozen holdout (ml/holdout.py). No "
        "experiment\n"
        "** may train or score on them before the final report.\n"
        "** Usual cause: a data/features.csv built before the freeze. "
        "Rebuild it:\n"
        "**     python ml/features/build.py\n")


def out_of_order(df: pd.DataFrame) -> tuple[int, int]:
    """(pairs whose OLDER version was published after the newer one, and
    how many of those straddle the boundary).

    Chains are diffed in version order. When a maintainer ships a backport
    after a newer major, 1.9.5 lands between 1.9.4 and 2.0.0 in that order
    while arriving on PyPI weeks later, and the pair 1.9.5 -> 2.0.0 is
    dated by 2.0.0. If 1.9.5 came out on or after the boundary and 2.0.0
    before it, that pair sits in dev with a post-boundary release at one
    end. Counted, so the real number replaces a guess. A version's date
    is known here only where it is some pair's version_to, so this is a
    lower bound.
    """
    when = released(df)
    pairs = (df[GROUP].assign(_to=when)
             .groupby(GROUP, dropna=False, sort=False)["_to"].max()
             .reset_index())
    dated = pairs.set_index(["package", "version_to"])["_to"]
    dated = dated[~dated.index.duplicated()]
    frm = dated.reindex(pd.MultiIndex.from_arrays(
        [pairs["package"], pairs["version_from"]])).to_numpy()
    frm = pd.Series(frm, index=pairs.index)
    late = frm.gt(pairs["_to"])
    across = late & frm.ge(HOLDOUT_START) & pairs["_to"].lt(HOLDOUT_START)
    return int(late.sum()), int(across.sum())


def fingerprint(df: pd.DataFrame) -> str:
    """Twelve hex characters naming the SET of version pairs.

    No labels and no features go into it, only which upgrades are in the
    set. Recorded when the holdout is frozen and printed again when it is
    opened, so "is this the holdout we froze?" is a comparison of two
    short strings rather than a matter of trust.
    """
    keys = sorted({"|".join(map(str, k))
                   for k in df[GROUP].itertuples(index=False, name=None)})
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]


def track(holdout: pd.DataFrame, write: bool = False) -> tuple[str, dict]:
    """How the holdout's SET OF PAIRS has moved since the day it froze.

    The rule is a date, so membership can move for legitimate reasons: a
    pipeline fix removes rows (F1 drops version strings, F16 collapses
    echoes) and can empty a pair; the F19 retry of the 149 crashed
    packages adds pairs. §15.4 is what happens when a dataset moves and
    nobody is watching, so this watches.

    build.py calls it with write=True: the first time, it saves the list
    to data/holdout_manifest.csv and never touches that file again.
    final_eval.py only reads it, so the frozen list cannot be created
    at the end.
    """
    now = holdout[GROUP].astype(str).drop_duplicates()
    have = set(now.itertuples(index=False, name=None))
    if not MANIFEST.exists():
        if write and have:
            now.sort_values(GROUP).to_csv(MANIFEST, index=False)
            return (f"  pair list saved to {MANIFEST} ({len(have):,} pairs). "
                    "That file IS the\n  frozen list: keep it. Later builds "
                    "report every change against it.",
                    {"frozen_pairs": len(have), "kept": len(have),
                     "gone": 0, "added": 0})
        return (f"  no {MANIFEST}, so how the holdout moved since the freeze "
                "cannot be said.", {})
    frozen = pd.read_csv(MANIFEST, dtype=str)[GROUP]
    was = set(frozen.itertuples(index=False, name=None))
    moved = {"frozen_pairs": len(was), "kept": len(was & have),
             "gone": len(was - have), "added": len(have - was)}
    if not moved["gone"] and not moved["added"]:
        return (f"  all {len(was):,} frozen pairs present, none added: the "
                "set is as frozen.", moved)
    return (f"  since the freeze: {moved['kept']:,} of {len(was):,} frozen "
            f"pairs still here, {moved['gone']:,} gone, {moved['added']:,} "
            "added.\n  Gone = emptied by a pipeline fix; added = new "
            "packages or later releases.\n  Both are allowed. Say which "
            "fix did it when the numbers are quoted.", moved)


def summary(holdout: pd.DataFrame,
            labels=("label", "label_scoped", "label_alias")) -> str:
    """What build.py prints about the holdout: its size, and whether it
    can be measured at the end.

    Counts of positives and of rankable pairs, never a model's score on
    it. Knowing how many positives the holdout holds cannot bias a model;
    knowing how well a model does on it can.
    """
    from ml.model.metrics import n_rankable  # sklearn; only needed here

    if holdout.empty:
        return (f"HOLDOUT: EMPTY. Nothing in this data was released on or "
                f"after {HOLDOUT_START.date()}.\nOlder datasets are fine for "
                "historical re-runs, but there is nothing here to seal.")

    when = released(holdout)
    lines = [
        f"HOLDOUT, frozen {FROZEN_ON}: every pair released on or after "
        f"{HOLDOUT_START.date()}",
        f"  {len(holdout):,} rows   "
        f"{len(holdout[GROUP].drop_duplicates()):,} version pairs   "
        f"{holdout['package'].nunique():,} packages   "
        f"{when.min().date()} to {when.max().date()}",
        f"  fingerprint {fingerprint(holdout)}   (names the set of pairs; "
        "no labels in it)",
        "",
        "  Can it be measured at the end? Same gates as stability.py:",
        f"  {MIN_POSITIVES}+ positives, {MIN_RANKABLE_PAIRS}+ rankable pairs.",
    ]
    short = []
    for lab in labels:
        if lab not in holdout:
            continue
        pos = int(holdout[lab].sum())
        r10, r20 = n_rankable(holdout, lab, 10), n_rankable(holdout, lab, 20)
        ok = (pos >= MIN_POSITIVES, r10 >= MIN_RANKABLE_PAIRS,
              r20 >= MIN_RANKABLE_PAIRS)
        lines.append(
            f"    {lab:<13} {pos:>5} positives   rankable pairs: "
            f"{r10:>3} at 10, {r20:>3} at 20   "
            + ("measurable" if all(ok) else "NOT FULLY MEASURABLE"))
        if not ok[0]:
            short.append(f"{lab}: PR-AUC rests on {pos} positives")
        if not ok[1]:
            short.append(f"{lab}: precision@10 would average {r10} upgrades")
        if not ok[2]:
            short.append(f"{lab}: nDCG@20 would average {r20} upgrades")
    if short:
        lines += ["", "  ** Below a gate: " + "; ".join(short) + ".",
                  "  ** Decide NOW whether to live with that (report those "
                  "metrics with their n,",
                  "  ** flagged) or move the boundary earlier. After today, "
                  "moving it is a",
                  "  ** choice made with the fixes' results in view, which is "
                  "the thing this",
                  "  ** file exists to stop."]
    return "\n".join(lines)
