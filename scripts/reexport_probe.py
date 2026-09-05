"""
How much is the exact-symbol join actually costing us?

    python scripts/reexport_probe.py

The evidence that prompted this, from label_check.py table 5:

    pygments   242 symbols used by someone,  192 changed,  0 overlap
    sympy       48 symbols used by someone,  895 changed,  0 overlap
    numba       53 symbols used by someone,  107 changed,  0 overlap
    litellm    424 symbols used by someone,  161 changed,  0 overlap

Zero overlap in every case, and these are libraries the ecosystem plainly
uses. The reason is visible in the examples: downstream code writes

    from sympy import Symbol          ->  recorded as sympy.Symbol
    from numba import njit            ->  recorded as numba.njit
    import litellm; litellm.completion->  recorded as litellm.completion

while griffe reports the change where the object is DEFINED:

    sympy.core.symbol.Symbol
    numba.core.decorators.njit
    litellm.main.completion

Same object, two different paths, and `==` says no. This is not an edge
case — it is what every well-organised package does, so the join fails
hardest exactly where usage is highest.

The (root, leaf) relaxation would catch all of these. It also produces
nonsense on packages with repetitive APIs: every google.cloud client class
carries DEFAULT_MTLS_ENDPOINT, so one leaf name matches dozens of unrelated
symbols and each gets credited with all 74 users.

So the question is not "relaxed or strict". It is whether the relaxation
can be SCOPED to the cases where the leaf name is unambiguous. This script
measures that, separating three populations:

    already matched  — the exact join found it
    unambiguous      — one changed symbol in the package owns this leaf
    ambiguous        — several do; crediting any of them is a guess

Nothing here writes labels. It tells you whether the fix is worth the
re-ingest before you spend the 35 minutes.
"""

import pathlib
import sys

import pandas as pd

DATA = pathlib.Path("data")
CHANGES = DATA / "changes.csv"
USAGE = DATA / "usage.csv"

# Libraries label_check flagged as used-by-the-world but never matched.
WATCH = ["pygments", "numba", "sympy", "litellm", "Cython", "docx",
         "pptx", "databricks", "dbt", "sglang"]


def split_parts(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    return s.str.split(".").str[0], s.str.split(".").str[-1]


def main() -> None:
    for p in (CHANGES, USAGE):
        if not p.exists():
            sys.exit(f"{p} not found — run the pipeline first.")

    ch = pd.read_csv(CHANGES)
    us = pd.read_csv(USAGE)

    ch["root"], ch["leaf"] = split_parts(ch["symbol"])
    us["root"], us["leaf"] = split_parts(us["symbol"])

    exact = set(us["symbol"])
    ch["exact_hit"] = ch["symbol"].isin(exact)

    # How many DISTINCT changed symbols in this package claim this leaf?
    # One  -> the mapping is unambiguous, the relaxation is safe.
    # Many -> DEFAULT_MTLS_ENDPOINT territory, and a guess.
    owners = ch.groupby(["root", "leaf"])["symbol"].transform("nunique")
    ch["leaf_owners"] = owners

    # Usage side, summed over everything sharing the (root, leaf).
    by_name = us.groupby(["root", "leaf"])["user_count"].sum()
    ch["name_users"] = pd.MultiIndex.from_arrays(
        [ch["root"], ch["leaf"]]).map(by_name).fillna(0).astype(int)

    unmatched = ~ch["exact_hit"] & (ch["name_users"] > 0)
    unambiguous = unmatched & (ch["leaf_owners"] == 1)
    ambiguous = unmatched & (ch["leaf_owners"] > 1)

    n = len(ch)
    print(f"\n{n:,} changed rows\n")
    print(f"  already matched exactly      {int(ch.exact_hit.sum()):>6,}"
          f"   ({ch.exact_hit.mean():.2%})")
    print(f"  unmatched, leaf unambiguous  {int(unambiguous.sum()):>6,}"
          f"   ({unambiguous.mean():.2%})   <- the recoverable ones")
    print(f"  unmatched, leaf ambiguous    {int(ambiguous.sum()):>6,}"
          f"   ({ambiguous.mean():.2%})   <- guesses, leave alone")

    rate_now = ch["exact_hit"].mean()
    rate_scoped = (ch["exact_hit"] | unambiguous).mean()
    rate_all = (ch["exact_hit"] | unmatched).mean()
    print(f"\npositive rate  strict {rate_now:.2%}"
          f"   scoped {rate_scoped:.2%}"
          f"   everything {rate_all:.2%}")

    print("\n" + "=" * 70)
    print("  WOULD THE FAMOUS ZERO-POSITIVE PACKAGES BE RESCUED?")
    print("=" * 70)
    rows = []
    for root in WATCH:
        sub = ch[ch["root"] == root]
        if sub.empty:
            continue
        rows.append({
            "package": root,
            "rows": len(sub),
            "exact": int(sub["exact_hit"].sum()),
            "unambiguous": int((~sub["exact_hit"] &
                                (sub["name_users"] > 0) &
                                (sub["leaf_owners"] == 1)).sum()),
            "ambiguous": int((~sub["exact_hit"] &
                              (sub["name_users"] > 0) &
                              (sub["leaf_owners"] > 1)).sum()),
        })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 70)
    print("  THE RECOVERABLE ROWS — read these, they are the evidence")
    print("=" * 70)
    top = (ch[unambiguous].sort_values("name_users", ascending=False)
                          .drop_duplicates("symbol").head(15))
    for _, r in top.iterrows():
        print(f"  {r.name_users:>4} pkgs use {r.root}.{r.leaf:<24} "
              f"changed at {r.symbol}")

    print("\n" + "=" * 70)
    print("  THE AMBIGUOUS ROWS — read these too, they are why not to")
    print("=" * 70)
    bad = (ch[ambiguous].sort_values(["name_users", "leaf_owners"],
                                     ascending=False).head(6))
    for _, r in bad.iterrows():
        print(f"  {r.name_users:>4} pkgs, but {r.leaf_owners} different "
              f"symbols in {r.root} end in .{r.leaf}")

    print("\nIf the first list is full of names you recognise and the second"
          "\nis full of repeated boilerplate, the scoped relaxation is the"
          "\nhonest label and the exact join was measuring the wrong thing.")


if __name__ == "__main__":
    main()
