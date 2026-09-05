"""
Turn labelled rows into a feature matrix, with a temporal split.

    python ml/features/build.py

Reads  data/labelled.csv
Writes data/features.csv   the same rows, plus derived features and a
                           `split` column of "train" / "test"

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

DATA = pathlib.Path("data")
LABELLED = DATA / "labelled.csv"
OUT = DATA / "features.csv"

# Anything computed from the usage index. Never a feature.
LEAKY = {"user_count", "name_user_count", "scoped_user_count",
         "label", "label_scoped", "leaf_owners"}

# The features themselves. Every one is knowable from griffe + PyPI alone.
NUMERIC = [
    "module_depth",       # click.echo (1) vs click.parser._Opt.add (3)
    "name_length",        # long names tend to be obscure
    "package_rank",       # 1 = most downloaded. The popularity prior.
    "release_size",       # how many changes shipped together
    "package_churn",      # how many changes this package makes overall
]
BOOLEAN = [
    "is_private",         # contract rule: any _component, dunders excluded
    "is_dunder",          # __version__ and friends
    "in_dunder_all",      # the package exported it on purpose
    "is_top_level",       # click.echo, the kind people import directly
    "is_version_string",  # the 36%-of-positives problem, made explicit
    "has_sub_target",     # a parameter changed, not the whole symbol
]
CATEGORICAL = ["kind", "bump"]

VERSION_LEAVES = {"__version__", "__VERSION__", "version",
                  "version_tuple", "__version_tuple__", "VERSION"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # What the version number PROMISED. This is also the semver baseline's
    # entire model, which is the point: the gap between the promise and the
    # label is the thesis.
    df["bump"] = [bump_type(a, b)
                  for a, b in zip(df["version_from"], df["version_to"])]

    leaf = df["symbol"].str.rsplit(".", n=1).str[-1]
    df["is_version_string"] = leaf.isin(VERSION_LEAVES)

    df["has_sub_target"] = df.get("sub_target", "").fillna("").ne("")

    # Context features: a change is easier to notice in a release of three
    # than in a release of eight hundred.
    df["release_size"] = df.groupby(
        ["package", "version_from", "version_to"])["symbol"].transform("size")
    df["package_churn"] = df.groupby("package")["symbol"].transform("size")

    for c in BOOLEAN:
        df[c] = df[c].fillna(False).astype(bool).astype(int)
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
    df = add_features(pd.read_csv(LABELLED))
    df = temporal_split(df)
    audit(df)

    print(f"\ntemporal split at {df.attrs['cutoff'].date()}")
    for name, part in df.groupby("split"):
        for lab in ("label", "label_scoped"):
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


if __name__ == "__main__":
    main()
