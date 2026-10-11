"""
F39: how many rows are package metadata. Counts only: nothing is fitted
or scored.

    python scripts/metadata_count.py

Reads  data/features.csv and data/holdout.csv, as build.py wrote them
       (version strings already out, F1)
Writes nothing

Step 1 ran it on files built before F39, to count what step 2 would take
out (NOTES §32.2). Once build.py drops these rows (step 2, §32.3), it
finds none in what build.py writes, and says so: that is the check that
the drop happened.

WHY (NOTES §31.2). The demo story's release, cryptography 46.0.7 ->
47.0.0, is ranked with cryptography.__about__.__copyright__ first: a
copyright notice whose year changed, used by nobody. A changed metadata
string breaks nobody, for the same reason a changed version string does
not, and F1 took those out of the model (§23). F39 does the same here.

THE RULE (NOTES §32.1), pushed before this script first ran on real data:

    A row is package metadata when the last part of its symbol is one of
    the names in build.py's METADATA_LEAVES (15 fixed before the count,
    two added after it by the test below, §32.2) and griffe reports that
    its value changed (ATTRIBUTE_CHANGED_VALUE). Other changes to those
    names stay: a removed __author__ can break code that reads it.

    The rows go where F1 sends version strings: out before any feature
    is computed, from features.csv and holdout.csv alike; with no score,
    so they sort last on the site; and into the all-clear (F15) beside the
    version strings.

    A name joins after this count only if it is package metadata by the
    same test: it describes the package (who made it, its licence, its
    links, its release) and no code calls it. NOTES marks any such name as
    added after the count. Section 6 lists the candidates with no labels
    beside them, so their labels cannot steer that.

WHAT IT PRINTS
  1. Each name: the rows F39 takes out (dev and holdout), how many of the
     dev ones are used under label_alias, the other changes to the same
     name (which stay), and an example.
  2. The totals, and the upgrades that hold nothing else, which F39
     empties: after it, those releases changed nothing anyone could feel.
  3. Where the dev rows sit: the single split's test half, and §31.1's
     story candidates (test half, public, a used change, 20+ changes).
  4. Why the model ranks them high: their reachability (§30.3) against
     every dev row's.
  5. The holdout: the rows and pairs it loses, and whether each label
     still clears the gates. Positives are counted only to know whether
     the holdout can be measured, which ml/holdout.py allows.
  6. For the record: every other dunder name whose value changed. Names,
     rows and an example only.
"""

import pathlib
import sys
import textwrap

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import (METADATA_KIND, METADATA_LEAVES,  # noqa: E402
                               VERSION_LEAVES, metadata_strings,
                               version_strings)
from ml.holdout import (GROUP, HOLDOUT_FILE, MIN_POSITIVES,  # noqa: E402
                        MIN_RANKABLE_PAIRS, assert_no_holdout)

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"
LABEL = "label_alias"
LABELS = ("label", "label_scoped", "label_alias")
STORY_MIN = 20   # §31.1: a story candidate has 20 or more public changes
SHOW = 30        # section 6 lists this many names, then a count of the rest


def leaves(df: pd.DataFrame) -> pd.Series:
    """The last part of each symbol: cryptography.__about__.__copyright__
    -> __copyright__."""
    return df["symbol"].astype(str).str.rsplit(".", n=1).str[-1]


def flag(s: pd.Series) -> pd.Series:
    """A 0/1 or True/False column as booleans, however it was written.
    bool("False") is True, so text is never cast directly (build.py)."""
    if s.dtype == object:
        return s.astype(str).str.strip().str.lower().isin({"true", "1"})
    return s.fillna(0).astype(bool)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    for f in (FEATURES, HOLDOUT_FILE):
        if not f.exists():
            sys.exit(f"{f} not found. Run python ml/features/build.py first.")
    keys = {c: str for c in GROUP}
    dev = pd.read_csv(FEATURES, dtype=keys)
    assert_no_holdout(dev, "metadata_count.py")
    held = pd.read_csv(HOLDOUT_FILE, dtype=keys)
    if LABEL not in dev:
        sys.exit(f"{FEATURES} has no {LABEL} column. Rebuild it: python "
                 "ml/features/labels.py, then ml/features/build.py.")
    if version_strings(dev).any() or version_strings(held).any():
        sys.exit("These files still hold version strings, so they were built "
                 "before F1 (item 2).\nRebuild them first: python "
                 "ml/features/build.py")
    return dev, held


def emptied(df: pd.DataFrame, gone: pd.Series) -> int:
    """Upgrades whose every row is going, so nothing is left in them."""
    pairs = len(df[GROUP].drop_duplicates())
    return pairs - len(df.loc[~gone, GROUP].drop_duplicates())


def gates(df: pd.DataFrame, label: str) -> tuple[int, int, int]:
    """(positives, pairs rankable at 10, pairs rankable at 20). Rankable:
    a pair with a positive and more than k changes, every key present, as
    scripts/holdout_boundary.py counts it."""
    p = (df.groupby(GROUP, sort=False, dropna=False)[label]
         .agg(rows="size", pos="sum").reset_index())
    keyed = p[GROUP].notna().all(axis=1)
    has = p["pos"] > 0
    return (int(p["pos"].sum()),
            int((keyed & has & (p["rows"] > 10)).sum()),
            int((keyed & has & (p["rows"] > 20)).sum()))


def candidates(dev: pd.DataFrame) -> pd.DataFrame:
    """§31.1's story candidates, found without a score: releases in the
    test half with 20 or more public changes, at least one of them used."""
    t = dev[dev["split"].eq("test") & ~flag(dev["is_private"])]
    g = (t.groupby(GROUP, sort=False)[LABEL]
         .agg(changes="size", used="sum").reset_index())
    return g[(g["used"] > 0) & (g["changes"] >= STORY_MIN)]


def example(rows: pd.DataFrame) -> str:
    """One row to show for a name: from the most downloaded package
    (package_rank 1 is the top), the first symbol on a tie."""
    rank = (pd.to_numeric(rows["package_rank"], errors="coerce")
            if "package_rank" in rows else pd.Series(0, index=rows.index))
    r = rows.assign(_rank=rank.fillna(10**9))
    x = r.sort_values(["_rank", "symbol", "version_to"],
                      kind="mergesort").iloc[0]
    return (f"{x['symbol']}, {x['package']} {x['version_from']} -> "
            f"{x['version_to']}")


def per_name(dev: pd.DataFrame, held: pd.DataFrame) -> list[dict]:
    """Section 1, one entry per name in METADATA_LEAVES."""
    m_dev, m_held = metadata_strings(dev), metadata_strings(held)
    n_dev, n_held = leaves(dev), leaves(held)
    out = []
    for name in sorted(METADATA_LEAVES):
        d = dev[m_dev & n_dev.eq(name)]
        h = held[m_held & n_held.eq(name)]
        stays = pd.concat([dev.loc[~m_dev & n_dev.eq(name), "kind"],
                           held.loc[~m_held & n_held.eq(name), "kind"]])
        shown = d if len(d) else h
        out.append({
            "name": name, "dev": len(d), "used": int(d[LABEL].sum()),
            "holdout": len(h),
            "packages": int(pd.concat([d["package"], h["package"]])
                            .nunique()),
            "stays": stays.astype(str).value_counts().sort_index().to_dict(),
            "example": example(shown) if len(shown) else "",
        })
    return out


def other_dunders(dev: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    """Section 6: dunder names not on either list whose value changed.
    Only the symbol and kind columns are read, so no label can reach the
    table."""
    every = pd.concat([dev[["symbol", "kind"]], held[["symbol", "kind"]]],
                      ignore_index=True)
    name = leaves(every)
    dunder = (name.str.len().gt(4) & name.str.startswith("__")
              & name.str.endswith("__"))
    keep = (dunder & every["kind"].astype(str).eq(METADATA_KIND)
            & ~name.isin(METADATA_LEAVES | VERSION_LEAVES))
    t = (every[keep].assign(name=name[keep])
         .groupby("name")["symbol"].agg(rows="size", example="min")
         .reset_index())
    return t.sort_values(["rows", "name"], ascending=[False, True],
                         kind="mergesort").reset_index(drop=True)


def pct(part: int, whole: int) -> str:
    return f"{part / whole:.2%}" if whole else "-"


def main() -> None:
    clash = METADATA_LEAVES & VERSION_LEAVES
    if clash:
        sys.exit(f"METADATA_LEAVES repeats F1's names: {sorted(clash)}. "
                 "Each row belongs to one fix.")
    dev, held = load()
    m_dev, m_held = metadata_strings(dev), metadata_strings(held)

    print("\nF39, COUNTS ONLY: nothing is fitted or scored.\n")
    print(f"  dev {len(dev):,} rows (features.csv), holdout {len(held):,} "
          "rows (holdout.csv),\n  version strings already out (F1)")
    print(f"  the rule (NOTES §32.1): the symbol ends in one of "
          f"{len(METADATA_LEAVES)} metadata names,\n  and griffe reports "
          f"{METADATA_KIND}")

    print("\n1. EACH NAME: rows F39 takes out, and other changes to it, "
          "which stay")
    quiet = []
    for e in sorted(per_name(dev, held),
                    key=lambda e: (-(e["dev"] + e["holdout"]), e["name"])):
        if not (e["dev"] or e["holdout"] or e["stays"]):
            quiet.append(e["name"])
            continue
        print(f"    {e['name']:<18} dev {e['dev']:>5,} ({e['used']:,} used)"
              f"   holdout {e['holdout']:>4,}   {e['packages']:,} "
              f"package{'' if e['packages'] == 1 else 's'}")
        if e["example"]:
            print(f"        e.g. {e['example']}")
        if e["stays"]:
            print("        stays: " + ", ".join(
                f"{k} {v:,}" for k, v in e["stays"].items()))
    if quiet:
        print(textwrap.fill("no change at all in this data: "
                            + ", ".join(quiet), width=78,
                            initial_indent="    ",
                            subsequent_indent="        "))

    n_dev, n_held = int(m_dev.sum()), int(m_held.sum())
    used, used_all = int(dev.loc[m_dev, LABEL].sum()), int(dev[LABEL].sum())
    print("\n2. IN ALL")
    print(f"  dev       {n_dev:,} of {len(dev):,} rows "
          f"({pct(n_dev, len(dev))}); {used:,} of the {used_all:,} used "
          f"rows under {LABEL}")
    print(f"  holdout   {n_held:,} of {len(held):,} rows "
          f"({pct(n_held, len(held))})")
    print(f"  upgrades emptied, nothing else in them: dev "
          f"{emptied(dev, m_dev):,}, holdout {emptied(held, m_held):,}")
    if not (n_dev or n_held):
        print("\n  F39 finds no rows. Either step 2 has run (build.py already "
              "takes them out),\n  or no release in this data changed one "
              "of these names.")
    else:
        print("  After F39 an emptied release changed nothing anyone could "
              "feel: it joins\n  the releases F1 emptied, for the all-clear "
              "(F15).")

    test = dev["split"].eq("test")
    print("\n3. WHERE THE DEV ROWS SIT")
    print(f"  the single split's test half: {int((m_dev & test).sum()):,} "
          f"rows, {int(dev.loc[m_dev & test, LABEL].sum()):,} used")
    before, after = candidates(dev), candidates(dev[~m_dev])
    marked = dev.loc[m_dev & test].assign(_name=leaves(dev[m_dev & test]))
    found = marked.groupby(GROUP, sort=False)["_name"].agg(
        lambda s: ", ".join(sorted(set(s))))
    hit = [(k, found[k]) for k in
           before[GROUP].itertuples(index=False, name=None)
           if k in found.index]
    print(f"  story candidates (§31.1): {len(before)} now, {len(after)} "
          f"after F39; {len(hit)} of them hold a metadata row")
    for (pkg, vf, vt), what in hit:
        print(f"    {pkg} {vf} -> {vt}: {what}")

    print("\n4. WHY THE MODEL RANKS THEM HIGH: reachability (§30.3)")
    if n_dev:
        meta = dev[m_dev]
        print(f"  {'public_depth':<13}median "
              f"{meta['public_depth'].median():.0f} for these rows, "
              f"{dev['public_depth'].median():.0f} for every dev row")
        for col, what in (("has_export_path", "re-exported"),
                          ("in_dunder_all", "in __all__"),
                          ("is_top_level", "top level")):
            if col in dev:
                print(f"  {what:<13}{flag(meta[col]).mean():.1%} of these "
                      f"rows, {flag(dev[col]).mean():.1%} of every dev row")
    else:
        print("  nothing to show: no dev row matches")

    gone = emptied(held, m_held)
    print("\n5. THE HOLDOUT, counts only")
    print(f"  F39 takes {n_held:,} rows out of it; {gone:,} "
          + ("pair holds nothing else and goes" if gone == 1
             else "pairs hold nothing else and go")
          + ",\n  as F1's did (build.py reports them against the frozen "
          "list).")
    print(f"  gates: {MIN_POSITIVES}+ positives, {MIN_RANKABLE_PAIRS}+ pairs "
          "rankable at 10 and at 20.\n  Each label reads positives / "
          "rankable at 10 / rankable at 20:")
    for lab in [lab for lab in LABELS if lab in held]:
        b, a = gates(held, lab), gates(held[~m_held], lab)
        ok = (a[0] >= MIN_POSITIVES and a[1] >= MIN_RANKABLE_PAIRS
              and a[2] >= MIN_RANKABLE_PAIRS)
        print(f"    {lab:<13} now {b[0]}/{b[1]}/{b[2]}, after F39 "
              f"{a[0]}/{a[1]}/{a[2]}   "
              + ("clears them" if ok else "BELOW A GATE"))

    print("\n6. FOR THE RECORD: other dunder names whose value changed")
    print("  Not on either list. Names, rows and an example only, no "
          "labels. One joins\n  F39 only if it describes the package and "
          "no code calls it (NOTES §32.1).")
    t = other_dunders(dev, held)
    for _, r in t.head(SHOW).iterrows():
        print(f"    {r['name']:<26}{r['rows']:>6,}   e.g. {r['example']}")
    if len(t) > SHOW:
        rest = t.iloc[SHOW:]
        print(f"    and {len(rest):,} more names, {int(rest['rows'].sum()):,} "
              "rows between them")
    if t.empty:
        print("    none")

    print("\n  This script changes nothing. The rule is METADATA_LEAVES in\n"
          "  ml/features/build.py; step 2 (NOTES §32) takes these rows out "
          "there,\n  where F1 takes out version strings.")


if __name__ == "__main__":
    main()
