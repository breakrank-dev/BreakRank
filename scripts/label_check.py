"""
Sit with the labels for an hour. This is the hour.

    python scripts/label_check.py

labels.py prints a pass/fail sanity report. This asks the harder question:
WHY is the positive rate what it is, and is the number honest?

The rate came in at 2.7%, below the book's 3-10% band, and there are three
competing explanations. Only one of them is a problem:

  1. Rank dilution      — adding packages 300-500 adds rows nobody uses.
                          Expected. Not a bug. Measurable below.
  2. Re-export gap      — click.utils.LazyFile changed, downstream writes
                          `from click import LazyFile`. Real usage, missed
                          by the exact join. A measured undercount.
  3. Broken join        — symbols on both sides that should match and do
                          not. THIS would be a bug, and it hides behind
                          the other two.

Every table below separates them.
"""

import pathlib
import sys

import pandas as pd

DATA = pathlib.Path("data")
LABELLED = DATA / "labelled.csv"


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def by_rank(df: pd.DataFrame) -> None:
    """Does the positive rate fall as packages get less popular?

    If yes, a lower rate on a bigger dataset is the dataset working as
    intended, not the labels breaking. If the rate is FLAT across ranks,
    the dilution story is wrong and something else is going on.
    """
    rule("1. POSITIVE RATE BY PACKAGE POPULARITY")
    if "package_rank" not in df.columns:
        print("no package_rank column — skipping")
        return

    d = df.dropna(subset=["package_rank"]).copy()
    d["bucket"] = (d["package_rank"] // 100 * 100).astype(int)
    t = d.groupby("bucket").agg(
        rows=("label", "size"),
        positives=("label", "sum"),
        rate=("label", "mean"),
    )
    t["rate"] = (t["rate"] * 100).round(2)
    t.index = [f"{b + 1}-{b + 100}" for b in t.index]
    print(t.to_string())

    if len(t) > 1:
        top, bottom = t["rate"].iloc[0], t["rate"].iloc[-1]
        print(f"\ntop bucket {top:.2f}%  vs  bottom bucket {bottom:.2f}%")
        if top > bottom * 1.3:
            print("-> Rate falls with popularity, as it must: a package"
                  "\n   nobody imports cannot have used symbols. The headline"
                  "\n   rate is an average over a deliberately long tail, and"
                  "\n   comparing it to a rate measured on the top 300 is"
                  "\n   comparing two different populations.")
        else:
            print("-> Rate is FLAT across popularity. That contradicts the"
                  "\n   dilution explanation — investigate the join instead.")


def version_churn(df: pd.DataFrame) -> None:
    """How much of the dataset is __version__ being bumped?"""
    rule("2. WHAT FRACTION IS PURE VERSION CHURN?")
    leaf = df["symbol"].str.rsplit(".", n=1).str[-1]
    is_v = leaf.isin({"__version__", "__VERSION__", "version",
                      "version_tuple", "__version_tuple__"})
    n = int(is_v.sum())
    print(f"{n:,} rows ({n / len(df):.1%}) are a version string changing value.")
    if n:
        print(f"positive rate among them: {df[is_v]['label'].mean():.1%}")
    print(f"\nall dunder rows: {int(df['is_dunder'].sum()):,} "
          f"({df['is_dunder'].mean():.1%})")
    print("\nThese are real changes and real usage — packages do read"
          "\n__version__ — but almost none of them BREAK anything. They are"
          "\nthe clearest case for why 'did it change' and 'does it matter'"
          "\nare different questions, which is the whole project.")

    without = df[~is_v]
    if len(without):
        print(f"\npositive rate excluding version churn: "
              f"{without['label'].mean():.2%}  "
              f"(overall: {df['label'].mean():.2%})")


def reexport_gap(df: pd.DataFrame) -> None:
    """The undercount we can measure but have not yet corrected."""
    rule("3. THE RE-EXPORT GAP")
    if "name_user_count" not in df.columns:
        print("no name_user_count column — skipping")
        return

    missed = (df["user_count"] == 0) & (df["name_user_count"] > 0)
    n = int(missed.sum())
    strict = df["label"].mean()
    relaxed = ((df["user_count"] > 0) | missed).mean()

    print(f"strict label (exact symbol match):      {strict:.2%}")
    print(f"if (root, leaf) matches counted too:    {relaxed:.2%}")
    print(f"rows that would flip 0 -> 1:            {n:,}")
    print("\nThe true rate is somewhere between these two. Strict undercounts"
          "\n(it misses `from click import LazyFile`); relaxed overcounts (two"
          "\nunrelated symbols in one package can share a leaf name). Neither"
          "\nnumber is 'the answer' — the week-4 experiment is to relabel with"
          "\nthe relaxed rule, retrain, and see which produces a better ranker."
          "\nThat is a measurement, not an opinion, and it is worth saying so.")

    if n:
        print("\nBiggest misses — check a few by hand, they are the evidence:")
        top = (df[missed].sort_values("name_user_count", ascending=False)
                        .drop_duplicates("symbol").head(6))
        for _, r in top.iterrows():
            print(f"  {int(r.name_user_count):>4} pkgs by name   {r.symbol}")


def concentration(df: pd.DataFrame) -> None:
    """Is the signal one package wearing a trenchcoat?"""
    rule("4. WHERE DO THE POSITIVES ACTUALLY COME FROM?")
    pos = df[df["label"] == 1]
    if pos.empty:
        print("no positives at all — the join is broken")
        return

    top = pos["package"].value_counts().head(8)
    share = top.iloc[0] / len(pos)
    print(f"{len(pos):,} positive rows across {pos['package'].nunique()} packages\n")
    for pkg, n in top.items():
        print(f"  {n:>5}  ({n / len(pos):>4.0%})  {pkg}")

    print()
    if share > 0.30:
        print(f"** {top.index[0]} alone is {share:.0%} of every positive row. **"
              "\nAny model trained on this learns that package, not the problem."
              "\nSay this number out loud before quoting any other number.")
    else:
        print(f"Largest single contributor is {share:.0%} — no package dominates,"
              "\nso the signal is not one library in disguise.")


def sanity_joins(df: pd.DataFrame) -> None:
    """The check that would catch a genuinely broken join."""
    rule("5. IS THE JOIN ITSELF SOUND?")
    roots = df["symbol"].str.split(".").str[0]
    per_root = df.groupby(roots)["label"].agg(["size", "sum"])
    dead = per_root[(per_root["sum"] == 0) & (per_root["size"] >= 100)]

    print(f"{df['symbol'].nunique():,} distinct symbols, "
          f"{roots.nunique()} module roots\n")
    if len(dead):
        print(f"{len(dead)} roots have 100+ rows and ZERO positives:")
        print(dead.sort_values("size", ascending=False).head(10).to_string())
        print("\nSome of these are honest (nobody imports them). But a root you"
              "\nRECOGNISE in this list is a red flag: it means the usage scan"
              "\nnever recorded anyone importing a famous library, which points"
              "\nat the import-name mapping, not at the ecosystem.")
    else:
        print("No large root is entirely unused. The join is doing its job.")


def main() -> None:
    if not LABELLED.exists():
        sys.exit(f"{LABELLED} not found — run ml/features/labels.py first.")
    df = pd.read_csv(LABELLED)

    print(f"\n{len(df):,} rows  |  {int(df['label'].sum()):,} positive "
          f"({df['label'].mean():.2%})  |  {df['package'].nunique()} packages")

    by_rank(df)
    version_churn(df)
    reexport_gap(df)
    concentration(df)
    sanity_joins(df)

    print("\n" + "=" * 70)
    print("  Write the answers into docs/NOTES.md while they are fresh.")
    print("  An examiner will ask why the rate is what it is, and 'I")
    print("  measured it and here is the decomposition' is the answer.")
    print("=" * 70)


if __name__ == "__main__":
    main()
