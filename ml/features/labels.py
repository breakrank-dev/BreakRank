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

from ml.contract import is_private_symbol  # noqa: E402  (frozen 3 Sep 2026)

DATA = pathlib.Path("data")
CHANGES = DATA / "changes.csv"
USAGE = DATA / "usage.csv"
OUT = DATA / "labelled.csv"


def root_leaf(symbol: str) -> tuple[str, str]:
    parts = symbol.split(".")
    return parts[0], parts[-1]


def _is_dunder_part(p: str) -> bool:
    return p.startswith("__") and p.endswith("__") and len(p) > 4


def add_labels(changes: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    df = changes.copy()

    # --- privacy flags, per the frozen API contract -----------------------
    # Day 3 settled by measurement that the original any-underscore flag was
    # two things wearing one name (all 161 "private" positives were dunders).
    # The API contract (change 1, 3 Sep 2026 — ml/contract.py) then froze
    # the shared rule, which adds one twist our Day-3 fix did not have:
    #   is_private -> any non-dunder _component, UNLESS the module's
    #                 __all__ exports the leaf name. A package that puts
    #                 _thing on its official signboard made it public.
    #   is_dunder  -> __names__ (ours, internal ML feature: __version__
    #                 churn is real usage but rarely a real break)
    # The __all__ membership itself (in_dunder_all) is captured at ingest,
    # because only the source on disk knows it; here we just apply it.
    parts_list = [s.split(".") for s in df["symbol"]]
    df["is_dunder"] = [any(_is_dunder_part(p) for p in parts) for parts in parts_list]
    df["is_private"] = [is_private_symbol(s) for s in df["symbol"]]
    if "in_dunder_all" in df.columns:
        exported = df["in_dunder_all"].fillna(False).astype(bool)
        df["is_private"] = df["is_private"] & ~exported
    else:
        print("note: this changes.csv predates the in_dunder_all column, so "
              "the __all__ override cannot apply yet. The next "
              "run_ingest.py --restart adds it.\n")

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

    # --- the SCOPED relaxation: a second label, measured not assumed -----
    # Downstream code writes `from pandas import read_csv`, recorded as
    # pandas.read_csv. griffe reports the change where the function is
    # DEFINED: pandas.io.parsers.readers.read_csv. Same function, two
    # paths, and the exact join says no — so the strict label throws away
    # 34 packages' worth of evidence about the single most famous symbol
    # in the dataset.
    #
    # The obvious fix — match on (root, leaf) — is too loose. Every
    # google.cloud client class carries DEFAULT_MTLS_ENDPOINT: 134 distinct
    # changed symbols share that one leaf, so the rule would credit all 74
    # users to each of them.
    #
    # So scope it. Relax ONLY where exactly one changed symbol in the
    # package owns that leaf name, which is precisely the case where the
    # re-export is unambiguous. Measured on 23,025 rows: recovers 462 rows
    # (2.65% -> 4.66%, into the book's expected band) and the recovered
    # rows are pandas.read_csv, httpx.get, pandas.concat, litellm.completion
    # — the examples the project exists to find.
    #
    # This is kept as a SECOND label, not a replacement. Which one trains a
    # better ranker is an experiment to run, not a thing to assert, and
    # having both columns is what makes the experiment possible.
    parts = df["symbol"].map(root_leaf)
    df["_root"] = [p[0] for p in parts]
    df["_leaf"] = [p[1] for p in parts]
    df["leaf_owners"] = df.groupby(["_root", "_leaf"])["symbol"].transform("nunique")

    unambiguous = (df["leaf_owners"] == 1) & (df["name_user_count"] > 0)
    df["label_scoped"] = ((df["user_count"] > 0) | unambiguous).astype(int)
    # What the scoped label would use as its count — the exact number when
    # we have one, the by-name number only where the mapping is certain.
    df["scoped_user_count"] = df["user_count"].where(
        df["user_count"] > 0, df["name_user_count"].where(unambiguous, 0))

    # --- the ALIAS label: the same join, done exactly ---------------------
    # label_scoped above is a HEURISTIC. It guesses that a change at
    # pandas.io.parsers.readers.read_csv is the same thing downstream code
    # calls pandas.read_csv, on the evidence that only one changed symbol
    # in pandas owns the leaf `read_csv`. That guess is usually right and
    # has no way of knowing when it is wrong.
    #
    # export_paths (added 6 Sep 2026, ml/ingest/api_extract.py) replaces the
    # guess with the fact. griffe knows the alias graph — `from .api import
    # get` is right there in the source — so the old version's aliases are
    # walked at ingest and every path downstream code could write is
    # recorded on the row. No leaf matching, no ambiguity, no 134 symbols
    # sharing DEFAULT_MTLS_ENDPOINT.
    #
    # MAX, not SUM, across a symbol's alias paths. usage.csv stores counts,
    # not the package names behind them, so summing `attr.attrib` and
    # `attrs.attrib` would double-count any package that imports both.
    # Max is the honest lower bound; the true figure needs a union we
    # cannot compute from this file.
    if "export_paths" in df.columns:
        paths = df["export_paths"].fillna("").astype(str)
        df["alias_user_count"] = [
            max([count] + [exact.get(p, 0) for p in s.split(";") if p])
            for count, s in zip(df["user_count"], paths)
        ]
        df["has_export_path"] = paths.ne("")

        # PUBLIC DEPTH — a feature, and the direct answer to NOTES §5.3.
        #
        # module_depth measures where a symbol is DEFINED. NOTES §5.3 showed
        # it was substantially predicting our own broken join: deep paths
        # scored zero not because deep symbols are unimportant but because
        # the exact match failed on them. pandas.io.parsers.readers.read_csv
        # has module_depth 4 and is one of the most used functions alive.
        #
        # public_depth measures where a symbol can be REACHED — the shortest
        # name a user can write for it. For read_csv that is 1, not 4. This
        # is what module_depth was always trying to be, and it is computed
        # from the package's own source with no downstream data anywhere
        # near it, so it is a feature and not a leak.
        df["public_depth"] = [
            min([d] + [p.count(".") for p in s.split(";") if p])
            for d, s in zip(df["module_depth"], paths)
        ]
    else:
        print("note: this changes.csv predates the export_paths column, so "
              "label_alias falls back to the exact join. Rerun "
              "ml/ingest/run_ingest.py --restart to populate it.\n")
        df["alias_user_count"] = df["user_count"]
        df["has_export_path"] = False
        df["public_depth"] = df["module_depth"]

    df["label_alias"] = (df["alias_user_count"] > 0).astype(int)

    return df.drop(columns=["_root", "_leaf"])


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
    if priv > 0.02:
        print(f"** {priv:.1%} of private-symbol rows are positive — investigate. **")

    print("\nDunder (__name__) rows, tracked separately since Day 3:")
    if df.is_dunder.any():
        print(f"  {int(df.is_dunder.sum()):,} rows, "
              f"{df[df.is_dunder]['label'].mean():.1%} positive — mostly "
              f"__version__ churn: really used, rarely breaking. The model "
              f"sees is_dunder explicitly and can learn to discount it.")

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

    if "label_scoped" in df:
        gained = int((df["label_scoped"] > df["label"]).sum())
        print(f"\nSCOPED label (relax only where one symbol owns the leaf):")
        print(f"  {int(df['label_scoped'].sum()):,} positive "
              f"({df['label_scoped'].mean():.2%})  —  {gained:,} rows recovered")
        print("  Both labels are written to the CSV. Train on each and compare;")
        print("  scripts/reexport_probe.py shows which rows the difference is.")

    if "label_alias" not in df:
        return

    print("\n" + "=" * 66)
    print("  THE THREE LABELS, SIDE BY SIDE")
    print("=" * 66)
    for name in ("label", "label_scoped", "label_alias"):
        print(f"  {name:<14} {int(df[name].sum()):>6,} positive "
              f"({df[name].mean():>6.2%})")

    if "has_export_path" in df:
        cov = df["has_export_path"].mean()
        print(f"\n{int(df['has_export_path'].sum()):,} rows ({cov:.1%}) have at "
              "least one shorter public name.")
        print("Rows without one are not failures: most symbols are simply never")
        print("re-exported, and packaging.utils.BuildTag really is the only")
        print("way to import it.")

    alias_only = int((df["label_alias"] > df["label"]).sum())
    print(f"\nRows the EXACT join missed and the alias graph recovered: "
          f"{alias_only:,}")

    # The comparison that matters. label_scoped guessed; label_alias knows.
    # Where they disagree, one of them is wrong, and which one tells us
    # whether NOTES section 5.3 was measuring the ecosystem or our own
    # broken join.
    scoped_only = (df["label_scoped"] == 1) & (df["label_alias"] == 0)
    alias_wins = (df["label_alias"] == 1) & (df["label_scoped"] == 0)
    print("\nWHERE THE HEURISTIC AND THE FACT DISAGREE")
    print(f"  scoped says used, alias graph says no: {int(scoped_only.sum()):,}")
    print("    ^ candidate FALSE POSITIVES of the leaf-name heuristic — the")
    print("      thing Varad's decision-10 worried about, now countable.")
    print(f"  alias graph says used, scoped says no:  {int(alias_wins.sum()):,}")
    print("    ^ real usage the heuristic was too cautious to claim.")

    if scoped_only.any():
        print("\n  A sample of the disputed rows:")
        cols = ["symbol", "name_user_count", "leaf_owners"]
        for _, r in df[scoped_only].head(5)[cols].iterrows():
            print(f"    {r.symbol}")
            print(f"      leaf used by {int(r.name_user_count)} pkgs, "
                  f"{int(r.leaf_owners)} symbol owns that leaf, "
                  "no alias path exists")

    print("\nWhich label ships is an experiment, not an opinion. Train on all")
    print("three and compare lift over the SAME-label baseline — PR-AUC is not")
    print("comparable across labels, because its floor is the positive rate.")


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
