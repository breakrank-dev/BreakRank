"""
The join that makes this a machine-learning project.

    python ml/features/labels.py

Reads  data/changes.csv   Track A: what changed        (15,000 rows, no labels)
       data/usage.csv     Track B: what the ecosystem uses
Writes data/labelled.csv  the training data

The label is one honest question:

    Does any downstream package import or reference this exact symbol?

        user_count > 0   ->   label = 1   (someone would feel this change)
        user_count = 0   ->   label = 0   (nobody could)

No hand labelling anywhere. This is called distant supervision, and being
able to say the words "my label is a proxy, and here is exactly where the
proxy is wrong" is worth more in an interview than the model itself. The
two known gaps, found on Day 1 and Day 3, both recorded before anyone asks:

  * Deprecation shims: click removed LazyFile from the source but serves it
    through __getattr__, so importers still work. Usage says "depended on",
    reality says "nothing broke yet". Label 1, truth (for now) 0.
  * Re-exports: a change row says click.utils.LazyFile; downstream code
    writes `from click import LazyFile`, which records click.LazyFile.
    Exact matching misses that. The name_user_count column measures the
    same-root-same-leaf relaxation, so week 4 can test whether relabelling
    with it beats the strict version — measured, not assumed.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

DATA = pathlib.Path("data")
CHANGES = DATA / "changes.csv"
USAGE = DATA / "usage.csv"
OUT = DATA / "labelled.csv"


def root_leaf(symbol: str) -> tuple[str, str]:
    parts = symbol.split(".")
    return parts[0], parts[-1]


def add_labels(changes: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    df = changes.copy()

    # --- exact match: the label ------------------------------------------
    exact = dict(zip(usage["symbol"], usage["user_count"]))
    df["user_count"] = df["symbol"].map(exact).fillna(0).astype(int)
    df["label"] = (df["user_count"] > 0).astype(int)

    # --- relaxed match: a feature, not the label -------------------------
    # click.utils.LazyFile and click.LazyFile share (root, leaf). Summing
    # user counts over that key catches re-exported usage at the cost of
    # some false matches between same-named symbols in one package.
    usage_rl = usage.copy()
    usage_rl[["root", "leaf"]] = usage_rl["symbol"].map(root_leaf).apply(pd.Series)
    by_name = usage_rl.groupby(["root", "leaf"])["user_count"].sum().to_dict()
    df["name_user_count"] = [
        by_name.get(root_leaf(s), 0) for s in df["symbol"]
    ]
    return df


def sanity_report(df: pd.DataFrame) -> None:
    """The checks from the book, section 6.5. Run them EVERY time."""
    n, pos = len(df), int(df["label"].sum())
    rate = pos / n if n else 0.0

    print("=" * 66)
    print(f"  {n:,} rows labelled  ->  {OUT}")
    print(f"  positive: {pos:,}  ({rate:.1%})")
    print("=" * 66)

    print("\nExpected 3-10% positive.", end=" ")
    if 0.03 <= rate <= 0.10:
        print("You are in range.")
    elif rate < 0.03:
        print(f"{rate:.1%} is LOW. Likely causes: usage scan covered too few "
              "packages, or re-exports (see name_user_count) hide real usage.")
    else:
        print(f"{rate:.1%} is HIGH. Likely causes: self-use leaking in, or "
              "call sites being counted instead of packages.")

    print("\nPositive rate by kind of change:")
    print((df.groupby("kind")["label"].mean().sort_values(ascending=False) * 100)
          .round(1).to_string())

    print("\nTHE sanity check — positive rate for private vs public symbols:")
    print((df.groupby("is_private")["label"].mean() * 100).round(2).to_string())
    priv = df[df.is_private]["label"].mean() if df.is_private.any() else 0
    print("\nPrivate symbols should be NEAR ZERO. Nobody imports _internals" )
    print("on purpose, so if they show real usage, alias resolution is broken.")
    if priv > 0.05:
        print(f"** {priv:.1%} of private-symbol rows are positive — investigate. **")

    print("\nMost-depended-on symbols that CHANGED (your headline examples):")
    # One symbol can appear many times — griffe emits a row per changed
    # parameter — so dedupe for display or the list is one symbol 8 times.
    top = (df[df.label == 1]
           .sort_values("user_count", ascending=False)
           .drop_duplicates("symbol").head(8))
    for _, r in top.iterrows():
        print(f"  {r.user_count:>4} pkgs   {r.symbol}   "
              f"[{r.kind}, {r.version_from} -> {r.version_to}]")

    caught = int(((df["user_count"] == 0) & (df["name_user_count"] > 0)).sum())
    print(f"\nRows the exact join misses but the (root, leaf) relaxation would")
    print(f"catch: {caught:,}. That number is the re-export gap, measured.")


def main() -> None:
    for p in (CHANGES, USAGE):
        if not p.exists():
            sys.exit(f"{p} not found — run Track A (run_ingest) and "
                     "Track B (run_usage) first.")
    changes = pd.read_csv(CHANGES)
    usage = pd.read_csv(USAGE)

    df = add_labels(changes, usage)
    df.to_csv(OUT, index=False)
    sanity_report(df)

    print("\nSit with this table before you build anything on it. The book")
    print("gives it an hour, and the book is right.")


if __name__ == "__main__":
    main()
