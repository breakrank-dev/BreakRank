"""
Turn labelled rows into a feature matrix, with a temporal split.

    python ml/features/build.py

Reads  data/labelled.csv
Writes data/features.csv   every row released BEFORE the frozen holdout,
                           plus derived features and a `split` column of
                           "train" / "test"
       data/holdout.csv    the frozen holdout: pairs released on or after
                           HOLDOUT_START, same columns, split = "holdout".
                           No experiment reads it; ml/holdout.py says why.
       data/holdout_manifest.csv   the holdout's list of version pairs,
                           written by the FIRST build after the freeze and
                           never again; later builds report changes to it

TWO RULES, and breaking either one makes every number afterwards a lie.

RULE 1 — NOTHING DERIVED FROM USAGE MAY BE A FEATURE.

    user_count, name_user_count, scoped_user_count, label, label_scoped

These all come from the Track B scan, and the label IS `user_count > 0`.
Feed any of them to the model and it scores ~1.00 PR-AUC by reading the
answer off the page. That number would be the most exciting thing in the
project and it would mean nothing at all. The check at the bottom of this
file fails loudly rather than trusting anyone to remember.

The honest question is: given ONLY what griffe and PyPI can tell us about
a change — its kind, how deep the symbol sits, what the version number
promised, how popular the package is — can we predict whether the
ecosystem depends on it? Everything the model sees has to be knowable
before the usage scan runs, because at serving time it is.

RULE 2 — THE SPLIT IS TEMPORAL, NEVER RANDOM.

A random split puts pandas 2.1.3 in train and pandas 2.1.4 in test, and
the model learns "pandas rows look like this" rather than anything about
breaking changes. Worse, it is the exact opposite of how the thing gets
used: we predict for releases that have not happened yet. So train on
what shipped before a cutoff date, test on what shipped after, and accept
the lower score as the true one.

A package appearing in both halves is FINE and deliberate — in production
we score new releases of packages we already know. Splitting by package
instead would answer a different question ("does this generalise to
libraries we have never seen?"), which is worth asking later but is not
the product.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml.contract import bump_type  # noqa: E402
from ml.holdout import (GROUP, HOLDOUT_FILE, HOLDOUT_START,  # noqa: E402
                        out_of_order, released, split_off, summary, track)

DATA = pathlib.Path("data")
LABELLED = DATA / "labelled.csv"
OUT = DATA / "features.csv"

# Anything computed from the usage index. Never a feature.
#
# alias_user_count and label_alias joined the list on 6 Sep 2026. They are
# derived from usage.csv exactly like user_count is — the alias GRAPH is
# clean, but the moment it is used to look up a usage number the result is
# the answer, not a feature.
LEAKY = {"user_count", "name_user_count", "scoped_user_count",
         "alias_user_count", "label", "label_scoped", "label_alias",
         "leaf_owners"}

# The features themselves. Every one is knowable from griffe + PyPI alone.
NUMERIC = [
    "module_depth",       # click.echo (1) vs click.parser._Opt.add (3)
    # Where the symbol can be REACHED, not where it is defined: the
    # shortest name a user can write. pandas.io.parsers.readers.read_csv
    # is module_depth 4 and public_depth 1. NOTES §5.3 found module_depth
    # was largely predicting our broken join; this is the honest version
    # of the same idea, and it comes from the package's own alias graph
    # with no downstream data involved.
    "public_depth",
    # Blast radius INSIDE the library: how many classes inherit this exact
    # change. 0 for almost everything; 3,242 for the ModuleUtilsMixin
    # methods transformers 5.17.0 dropped. This column is what replaced
    # 9,729 duplicate rows (NOTES §10.2), and it is a feature rather than
    # bookkeeping because a method thousands of classes inherit is a
    # different kind of break from one on a leaf class. Knowable from
    # griffe alone — no downstream data touches it.
    "inherited_by",
    # How many breakages this module already produced EARLIER in the same
    # version chain. A module that churns constantly has users who expect
    # it; a first break in a quiet module is a different event. Written at
    # extract time because it needs the whole chain in order — see the
    # note in run_ingest.CHANGE_COLS.
    #
    # It is history, not usage: every number in it comes from griffe diffs
    # of releases that had already shipped when the pair being scored was
    # published. Nothing downstream, nothing from the future.
    "prior_breaks_in_module",
    "name_length",        # long names tend to be obscure
    "package_rank",       # 1 = most downloaded. The popularity prior.
    "release_size",       # how many changes shipped together
    # How many changes this package shipped in EARLIER releases. Past only
    # since F5 (item 2): it used to be the package's total over the whole
    # file, which a model could not know at serving time.
    "package_churn",
]
BOOLEAN = [
    "is_private",         # contract rule: any _component, dunders excluded
    "is_dunder",          # __init__ and friends; __version__ goes in F1
    "in_dunder_all",      # the package exported it on purpose
    "is_top_level",       # click.echo, the kind people import directly
    # is_version_string was here until F1 (item 2). The rows it flagged are
    # now dropped before any feature is computed (drop_version_strings),
    # so the flag would be False on every row that is left.
    "has_sub_target",     # a parameter changed, not the whole symbol
    "has_export_path",    # re-exported under a shorter public name
    # Did the OLD release carry a deprecation marker for this symbol — a
    # @deprecated decorator, the word in its docstring, or griffe's own
    # flag? The maintainer said "this is going away" and then it went
    # away. Measured on griffe 2.2.0, `griffe.deprecated` is None for BOTH
    # docstring markers and @deprecated decorators, so reading that flag
    # alone would have scored every row False and looked like a dead
    # feature rather than a broken reader. NOTES §16.1.
    "was_deprecated_before",
]
CATEGORICAL = ["kind", "bump"]

VERSION_LEAVES = {"__version__", "__VERSION__", "version",
                  "version_tuple", "__version_tuple__", "VERSION"}


def version_strings(df: pd.DataFrame) -> pd.Series:
    """True for rows whose symbol is a version constant (pkg.__version__,
    pkg.VERSION and the like), judged by the last part of its name."""
    leaf = df["symbol"].astype(str).str.rsplit(".", n=1).str[-1]
    return leaf.isin(VERSION_LEAVES)


def drop_version_strings(df: pd.DataFrame) -> pd.DataFrame:
    """F1 (item 2 of the fix list): the rows that are a version constant
    changing value, removed before any feature is computed.

    1,760 of the 1,769 such rows are ATTRIBUTE_CHANGED_VALUE on
    pkg.__version__ or VERSION. Downstream code reads the constant, so the
    label calls it used, and version strings were 45% of the positives
    while breaking nobody: they change on every release by design. The
    model's top feature was detecting them. Dropping them HERE, before
    add_features, is what keeps release_size and package_churn from
    counting them. labelled.csv keeps them, for the label reports that
    describe them.
    """
    return df[~version_strings(df)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # What the version number PROMISED. This is also the semver baseline's
    # entire model, which is the point: the gap between the promise and the
    # label is the thesis.
    df["bump"] = [bump_type(a, b)
                  for a, b in zip(df["version_from"], df["version_to"])]

    df["has_sub_target"] = df.get("sub_target", "").fillna("").ne("")

    # Written by the extractor (api_extract.fold_inherited). Defaulted here
    # rather than assumed present so a changes.csv from before 12 Sep still
    # builds — it scores every row 0, which is what "we did not measure
    # this" should look like, not a crash and not a silent NaN column.
    if "inherited_by" not in df.columns:
        df["inherited_by"] = 0
    df["inherited_by"] = (pd.to_numeric(df["inherited_by"], errors="coerce")
                          .fillna(0).astype(int))

    # THE HISTORY FEATURES DEFAULT, BUT THEY SAY SO.
    #
    # Same tolerance as inherited_by — an older changes.csv still builds —
    # with one difference that matters. A missing `inherited_by` defaults
    # to 0 and 0 is also its true value for most rows, so the default is
    # nearly harmless. A missing `was_deprecated_before` defaults to False
    # and False is a CLAIM: "the maintainer never warned anyone." If the
    # column is simply absent, that claim is made about every row in the
    # dataset, the feature reports zero gain, and the obvious reading is
    # "deprecation does not predict usage" — a finding, from a column that
    # was never written. So it warns rather than defaulting in silence.
    for col, default in (("was_deprecated_before", False),
                         ("prior_breaks_in_module", 0)):
        if col not in df.columns:
            print(f"** {col} is not in this file — every row will read "
                  f"{default!r}. Re-ingest before reporting on it. **")
            df[col] = default
    df["prior_breaks_in_module"] = (
        pd.to_numeric(df["prior_breaks_in_module"], errors="coerce")
        .fillna(0).astype(int))

    # Context features: a change is easier to notice in a release of three
    # than in a release of eight hundred.
    df["release_size"] = df.groupby(
        ["package", "version_from", "version_to"])["symbol"].transform("size")

    # PAST ONLY (F5, item 2). The changes this package shipped in releases
    # dated BEFORE this one, among the rows handed in. It used to be the
    # package's total over the whole file: a constant per package, a median
    # 33% of it from releases after the cut, and one of the three places
    # where the holdout reached dev rows (KNOWN CROSSINGS, ml/holdout.py).
    # Rows of one release share its date, so they share a count, and no
    # release counts itself. An undated row has no known past and reads 0.
    when = released(df)
    df["package_churn"] = (when.groupby(df["package"]).rank(method="min")
                           .sub(1).fillna(0).astype(int))

    # `.astype(bool)` ON A STRING COLUMN IS A TRAP. Python says
    # bool("False") is True — every non-empty string is truthy — so if a
    # boolean column ever arrives as text instead of pandas' inferred bool
    # dtype, this loop silently sets the whole column to 1 and the feature
    # becomes a constant. read_csv usually infers bool and usually it never
    # happens; "usually" is not a thing to leave in a cast that cannot
    # fail loudly. Map the text forms explicitly first.
    TRUE = {"true", "1", "yes", "t"}
    for c in BOOLEAN:
        col = df[c]
        if col.dtype == object:
            col = col.map(lambda v: v if isinstance(v, (bool, int, float))
                          else str(v).strip().lower() in TRUE)
        df[c] = col.fillna(False).astype(bool).astype(int)
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.25) -> pd.DataFrame:
    """Older releases train, newer releases test. Cut on a real date."""
    df = df.copy()
    d = pd.to_datetime(df["released_at"], errors="coerce")
    df["_when"] = d

    known = d.dropna()
    if known.empty:
        sys.exit("no usable released_at dates — cannot split temporally")

    cutoff = known.quantile(1 - test_frac)
    df["split"] = "train"
    df.loc[df["_when"] > cutoff, "split"] = "test"
    # Rows with no date cannot be placed in time; training on them is safe,
    # testing on them is not, so they go to train.
    df.loc[df["_when"].isna(), "split"] = "train"
    df.attrs["cutoff"] = cutoff
    return df.drop(columns=["_when"])


def audit(df: pd.DataFrame) -> None:
    features = NUMERIC + BOOLEAN + CATEGORICAL
    bad = LEAKY.intersection(features)
    if bad:
        sys.exit(f"LEAKAGE: {sorted(bad)} are derived from the usage index "
                 "and must never be features. Refusing to write.")

    missing = [c for c in features if c not in df.columns]
    if missing:
        sys.exit(f"missing feature columns: {missing}")

    print(f"\n{len(features)} features, none derived from usage:")
    print(f"  numeric      {', '.join(NUMERIC)}")
    print(f"  boolean      {', '.join(BOOLEAN)}")
    print(f"  categorical  {', '.join(CATEGORICAL)}")
    print(f"\nheld out of the model on purpose: {', '.join(sorted(LEAKY))}")


def main() -> None:
    if not LABELLED.exists():
        sys.exit(f"{LABELLED} not found — run ml/features/labels.py first.")
    raw = pd.read_csv(LABELLED)
    df = add_features(drop_version_strings(raw))
    gone = version_strings(raw)
    emptied = (len(raw[GROUP].drop_duplicates())
               - len(raw.loc[~gone, GROUP].drop_duplicates()))
    print(f"\nF1: {int(gone.sum()):,} version-string rows dropped before any "
          "feature was computed"
          + (f", {int(raw.loc[gone, 'label'].sum()):,} of them positive "
             "under label" if "label" in raw else "")
          + f".\n    {emptied:,} upgrades held nothing else and are gone.")

    # THE HOLDOUT COMES OFF BEFORE THE SPLIT, never after. The train/test
    # cut below is a quantile of whatever rows it is handed, so it has to
    # be handed the dev rows only, or the holdout would decide where the
    # dev split falls.
    #
    # Features are computed on the full frame first, as they would be at
    # serving time. That is correct for every feature that only looks
    # backwards in time, which since F5 includes package_churn. Two smaller
    # crossings remain, listed under KNOWN CROSSINGS in ml/holdout.py.
    undated = int(released(df).isna().sum())
    late, across = out_of_order(df)
    df, holdout = split_off(df)
    if df.empty:
        sys.exit(f"every dated row is on or after {HOLDOUT_START.date()}, "
                 "so nothing is left to train on.")
    df = temporal_split(df)
    audit(df)

    print(f"\ntemporal split at {df.attrs['cutoff'].date()}, over rows "
          f"released before {HOLDOUT_START.date()}")
    for name, part in df.groupby("split"):
        for lab in ("label", "label_scoped", "label_alias"):
            if lab in part:
                print(f"  {name:<5} {len(part):>7,} rows   "
                      f"{lab:<13} {int(part[lab].sum()):>5} positive "
                      f"({part[lab].mean():.2%})")
    print()

    test = df[df.split == "test"]
    if "label" in test and test["label"].sum() < 30:
        print(f"** only {int(test['label'].sum())} positives in test — "
              "metrics will be noisy. Consider a larger test_frac. **\n")

    groups = df.groupby(["package", "version_from", "version_to"]).ngroups
    print(f"{groups:,} distinct version pairs — precision@10 is measured "
          "PER PAIR,\nbecause 'which 10 of THIS upgrade's changes matter' is "
          "the actual product.")

    df.to_csv(OUT, index=False)
    print(f"\n{OUT}: {len(df):,} rows")

    holdout.assign(split="holdout").to_csv(HOLDOUT_FILE, index=False)
    print(f"{HOLDOUT_FILE}: {len(holdout):,} rows, sealed. Only "
          "ml/model/final_eval.py --unseal scores it\nagainst its labels; "
          "ml/db.py reads it to serve scores, nothing more.\n")
    print(summary(holdout))
    print()
    print(track(holdout, write=True)[0])

    print(f"\n  time order: {late} pair(s) diff a release against one "
          f"published after it,\n  {across} of them across the boundary "
          "(backports; see KNOWN CROSSINGS in ml/holdout.py).")
    if undated:
        print(f"  ** {undated:,} rows have no usable released_at. They "
              "train and are never tested\n  ** or held out; §21.7 counted "
              "none, so find out why before going on.")


if __name__ == "__main__":
    main()
