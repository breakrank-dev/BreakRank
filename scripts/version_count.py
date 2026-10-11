"""
F1 narrowed (NOTES §33): which rows the version-string and metadata rules
take out of the model, as they were until 10 Oct and as they are now, and
what that does to the all-clear. Counts only: nothing is fitted or scored.

    python scripts/version_count.py

Reads  data/labelled.csv: every row labels.py wrote, dev and holdout, with
       version strings and package metadata still in
       data/holdout_manifest.csv, if it is there (the frozen pair list)
Writes nothing

WHY (NOTES §33.1). F1 took out any change to six version names, judged by
the last part of the symbol (§23). Varad asked on 9 Oct for it to work the
way F39 does. A removed __version__ breaks every line that reads
pkg.__version__, so only a changed value goes. And version, VERSION and
version_tuple are also a submodule (packaging.version), a method or a
class's own attribute, so as bare names they count only where they sit
directly in a module. The pipeline decides the all-clear (his option (b)):
the loader stops writing these rows, a deletion step of its own removes
the ones already in the database, and a release with nothing else reads
analysed_clean. This says how many rows and releases that is.

THE RULE (NOTES §33.1), pushed before this script first ran on real data,
is version_strings() in ml/features/build.py: griffe reports
ATTRIBUTE_CHANGED_VALUE, and the symbol ends in __version__, __VERSION__
or __version_tuple__, or in version, VERSION or version_tuple where
griffe's explanation names it by that bare name, which griffe does only
for an attribute directly in a module. A row the explanation cannot place
stays in the model. So does a value griffe reports as changed to `unset`:
the value is gone and only an annotation is left, a removal in effect.
That last exception applies to F39's metadata names too (§33.1).

WHAT IT PRINTS
  1. The rules side by side, until 10 Oct and from it: rows in dev and in
     the holdout, the used ones in dev, and the upgrades each leaves with
     nothing in them.
  2. What comes back into the model, by name and by why: not a changed
     value, a value changed to unset, not directly in a module, or an
     explanation this cannot read.
  3. What still goes, by name, and every symbol behind the bare names, so
     anyone can check they are the package's own version.
  4. The all-clear: releases with nothing left once F1 and F39 have run,
     which read analysed_clean after the loader change and the deletion
     step; the ones that leave it; and the most downloaded packages' ones,
     for a preset on the site.
  5. What the loader stops writing. The deletion step counts again in the
     database itself, which also holds rows from earlier loads.
  6. The holdout, counts only: the rows and pairs that come back, its
     fingerprint, how it stands against the frozen list, and whether each
     label still clears the gates. Positives are counted only for the
     gates, which ml/holdout.py allows.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import (GRIFFE_TITLE, METADATA_KIND,  # noqa: E402
                               METADATA_LEAVES, UNSET, VERSION_BARE,
                               VERSION_DUNDERS, VERSION_KIND, leaves,
                               metadata_strings, version_strings)
from ml.holdout import (GROUP, MIN_POSITIVES,  # noqa: E402
                        MIN_RANKABLE_PAIRS, fingerprint, holdout_mask,
                        released, track)

LABELLED = pathlib.Path("data") / "labelled.csv"
LABEL = "label_alias"
LABELS = ("label", "label_scoped", "label_alias")
# F1 as it was until 10 Oct (§23): any change to these six names, judged by
# the last part of the symbol alone. Written out here because build.py no
# longer holds it.
OLD_LEAVES = {"__version__", "__VERSION__", "version", "version_tuple",
              "__version_tuple__", "VERSION"}
PRESETS = 10   # section 4 lists this many packages' all-clear releases
SHOWN = 12     # sections 2 and 4 list this many lines, then a count
SYMBOLS = 40   # section 3 lists this many symbols behind the bare names


def load() -> pd.DataFrame:
    if not LABELLED.exists():
        sys.exit(f"{LABELLED} not found. Run python ml/features/labels.py "
                 "first.")
    df = pd.read_csv(LABELLED, dtype={c: str for c in GROUP})
    missing = [c for c in ("symbol", "kind", "explanation", LABEL)
               if c not in df]
    if missing:
        sys.exit(f"{LABELLED} has no {', '.join(missing)} column. The rule "
                 "reads griffe's explanation,\nso this needs the file "
                 "labels.py writes from today's changes.csv.")
    return df


def old_f1(df: pd.DataFrame) -> pd.Series:
    """F1 until 10 Oct: any change to the six names."""
    return leaves(df).isin(OLD_LEAVES)


def old_f39(df: pd.DataFrame) -> pd.Series:
    """F39 until 10 Oct (§32.1): a changed value, `unset` included."""
    return (leaves(df).isin(METADATA_LEAVES)
            & df["kind"].astype(str).eq(METADATA_KIND))


def title(df: pd.DataFrame) -> pd.Series:
    """The object's path inside its module, as griffe's explanation gives
    it before the sentence for the kind; "" where it cannot be read."""
    return (df["explanation"].fillna("").astype(str)
            .str.extract(GRIFFE_TITLE, expand=False).fillna("").astype(str))


def why_back(df: pd.DataFrame) -> pd.Series:
    """For each row the old rules took and the new ones keep: why it stays.
    The example printed beside it shows where it sits."""
    t = title(df)
    other = df["kind"].astype(str).ne(VERSION_KIND)
    unset = (df["explanation"].fillna("").astype(str).str.rstrip()
             .str.endswith(UNSET))
    return pd.Series(
        ["not a changed value: " + str(k) if o
         else "value changed to unset: a removal in effect" if u
         else "griffe's explanation cannot be read" if not s
         else "not directly in a module"
         for k, o, u, s in zip(df["kind"], other, unset, t)],
        index=df.index)


def keyed(keys) -> list:
    """Upgrade keys in a stable order, whatever is missing from them."""
    return sorted(keys, key=lambda k: tuple(map(str, k)))


def emptied(df: pd.DataFrame, gone: pd.Series) -> int:
    """Upgrades whose every row is going, so nothing is left in them."""
    pairs = len(df[GROUP].drop_duplicates())
    return pairs - len(df.loc[~gone, GROUP].drop_duplicates())


def clear(df: pd.DataFrame, gone: pd.Series) -> set:
    """The upgrades with nothing left once `gone` is out."""
    left = set(df.loc[~gone, GROUP].itertuples(index=False, name=None))
    every = set(df[GROUP].itertuples(index=False, name=None))
    return every - left


def gates(df: pd.DataFrame, label: str) -> tuple[int, int, int]:
    """(positives, pairs rankable at 10, pairs rankable at 20), counted as
    scripts/holdout_boundary.py and metadata_count.py count them."""
    if df.empty:
        return 0, 0, 0
    p = (df.groupby(GROUP, sort=False, dropna=False)[label]
         .agg(rows="size", pos="sum").reset_index())
    has_keys = p[GROUP].notna().all(axis=1)
    has = p["pos"] > 0
    return (int(p["pos"].sum()),
            int((has_keys & has & (p["rows"] > 10)).sum()),
            int((has_keys & has & (p["rows"] > 20)).sum()))


def rank(df: pd.DataFrame) -> pd.Series:
    """package_rank as a number; 1 is the most downloaded, unknown last."""
    if "package_rank" not in df:
        return pd.Series(10**9, index=df.index)
    return pd.to_numeric(df["package_rank"], errors="coerce").fillna(10**9)


def example(rows: pd.DataFrame) -> str:
    """One row to show: from the most downloaded package, the first symbol
    on a tie."""
    x = (rows.assign(_rank=rank(rows))
         .sort_values(["_rank", "symbol", "version_to"], kind="mergesort")
         .iloc[0])
    return (f"{x['symbol']}, {x['package']} {x['version_from']} -> "
            f"{x['version_to']}")


def used(rows: pd.DataFrame, held: pd.Series) -> int:
    """Used rows under LABEL, counted in dev only."""
    return int(pd.to_numeric(rows.loc[~held.loc[rows.index], LABEL],
                             errors="coerce").fillna(0).sum())


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def presets(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    """The all-clear releases of the most downloaded packages, the newest
    one per package by its parsed date: a release the site can show as
    analysed_clean."""
    if not keys:
        return pd.DataFrame(columns=GROUP + ["_rank", "_when", "what", "n"])
    rows = df.merge(pd.DataFrame(list(keys), columns=GROUP), on=GROUP)
    rows = rows.assign(_rank=rank(rows), _leaf=leaves(rows),
                       _when=released(rows))
    per = (rows.groupby(GROUP, sort=False, dropna=False)
           .agg(_rank=("_rank", "min"), _when=("_when", "max"),
                what=("_leaf", lambda s: ", ".join(sorted(set(s)))),
                n=("symbol", "size"))
           .reset_index()
           .sort_values(["_rank", "_when", "version_to"],
                        ascending=[True, False, False], kind="mergesort",
                        na_position="last"))
    return per.drop_duplicates("package").head(PRESETS)


def main() -> None:
    if VERSION_DUNDERS | VERSION_BARE != OLD_LEAVES:
        sys.exit("build.py's version names are not the six F1 has always "
                 "had: "
                 f"{sorted(VERSION_DUNDERS | VERSION_BARE)}.\nNOTES §33.1 "
                 "narrowed WHERE they count, not WHICH names; fix build.py.")
    df = load()
    held = holdout_mask(df)
    f1_was, f1_now = old_f1(df), version_strings(df)
    m_was, m_now = old_f39(df), metadata_strings(df)
    old, new = f1_was | m_was, f1_now | m_now
    if (new & ~old).any():
        sys.exit(f"The new rules take {int((new & ~old).sum()):,} rows the "
                 "old ones did not. They can only take fewer.")
    back = old & ~new

    print("\nF1 NARROWED (NOTES §33), COUNTS ONLY: nothing is fitted or "
          "scored.\n")
    print(f"  {LABELLED}: {len(df):,} rows, {int((~held).sum()):,} in dev "
          f"and {int(held.sum()):,} in the holdout")
    print("  the rule (NOTES §33.1): griffe reports a value changed, not to "
          "unset, and the\n  symbol ends in "
          + ", ".join(sorted(VERSION_DUNDERS)) + " anywhere,\n  or in "
          + ", ".join(sorted(VERSION_BARE))
          + " where griffe's\n  explanation puts it directly in a module")

    print("\n1. THE RULES, until 10 Oct and from it (used: dev only, under "
          f"{LABEL})")
    print(f"  {'':<22}{'rows':>7}{'dev':>9}{'used':>7}{'holdout':>9}"
          f"{'upgrades emptied':>18}")
    for name, rule in (("F1 until 10 Oct", f1_was), ("F1 from 10 Oct", f1_now),
                       ("F39 until 10 Oct", m_was),
                       ("F39 from 10 Oct", m_now)):
        n = int(rule.sum())
        print(f"  {name:<22}{n:>7,}{int((rule & ~held).sum()):>9,}"
              f"{used(df[rule], held):>7,}{int((rule & held).sum()):>9,}"
              f"{emptied(df, rule):>18,}")
    for name, f1, both in (("until 10 Oct", f1_was, old),
                           ("from 10 Oct", f1_now, new)):
        e = emptied(df, both)
        print(f"  F1 and F39 {name} together empty {e:,} upgrades, "
              f"{e - emptied(df, f1):,} more than F1 alone")
    if not f1_was.any():
        print("\n  No row matches even the old rule, so these are not the "
              "files labels.py\n  writes: labelled.csv keeps version "
              "strings. Check the file before reading on.")

    print("\n2. WHAT COMES BACK INTO THE MODEL: the old rules took these, the "
          "new ones keep them")
    if back.any():
        b = df[back].assign(_name=leaves(df[back]), _why=why_back(df[back]))
        lines = []
        for (name, why), g in b.groupby(["_name", "_why"], sort=False):
            lines.append((len(g), name, why, int((~held[g.index]).sum()),
                          used(g, held), int(held[g.index].sum()),
                          example(g)))
        lines.sort(key=lambda x: (-x[0], x[1], x[2]))
        for n, name, why, d, u, h, ex in lines[:SHOWN]:
            print(f"    {name:<18}{why}")
            print(f"        dev {d:,} ({u:,} used), holdout {h:,}   e.g. {ex}")
        if len(lines) > SHOWN:
            rest = lines[SHOWN:]
            print(f"    and {len(rest):,} more lines, "
                  f"{sum(x[0] for x in rest):,} rows between them")
        print(f"  in all {int(back.sum()):,} rows: dev "
              f"{int((back & ~held).sum()):,} ({used(df[back], held):,} "
              f"used), holdout {int((back & held).sum()):,}")
    else:
        print("    none: the old and new rules take the same rows here")

    print("\n3. WHAT F1 STILL TAKES, by name")
    gone = df[f1_now].assign(_name=leaves(df[f1_now]))
    for name, g in sorted(gone.groupby("_name"), key=lambda x: -len(x[1])):
        print(f"    {name:<18}dev {int((~held[g.index]).sum()):>6,} "
              f"({used(g, held):,} used)   holdout "
              f"{int(held[g.index].sum()):>5,}   "
              + plural(g["package"].nunique(), "package"))
    bare = gone[gone["_name"].isin(VERSION_BARE)]
    if len(bare):
        sym = (bare.assign(_rank=rank(bare))
               .groupby(["symbol", "package"], sort=False)
               .agg(_rank=("_rank", "min"), n=("symbol", "size"))
               .reset_index()
               .sort_values(["_rank", "symbol"], kind="mergesort"))
        print(f"  Every symbol behind the bare names, so each can be checked "
              f"as the package's\n  own version ({len(sym):,} symbols, most "
              "downloaded package first):")
        for _, x in sym.head(SYMBOLS).iterrows():
            print(f"    {x['symbol']}   {plural(int(x['n']), 'release')}")
        if len(sym) > SYMBOLS:
            print(f"    and {len(sym) - SYMBOLS:,} more symbols, "
                  f"{int(sym['n'].iloc[SYMBOLS:].sum()):,} rows between them")
    if gone.empty:
        print("    none")

    print("\n4. THE ALL-CLEAR: releases with nothing left once F1 and F39 "
          "have run")
    was, now = clear(df, old), clear(df, new)
    in_held = set(df.loc[held, GROUP].itertuples(index=False, name=None))
    print(f"  until 10 Oct {len(was):,}, from 10 Oct {len(now):,} "
          f"({len(now - in_held):,} in dev, {len(now & in_held):,} in the "
          "holdout)")
    print("  After the loader change and the deletion step these read "
          "analysed_clean,\n  with n_changes 0.")
    left = keyed(was - now)
    if left:
        print(f"  {len(left):,} leave the all-clear, because a row in them "
              "now stays:")
        b = df[back]
        what = (b.assign(_d=leaves(b) + " " + b["kind"].astype(str))
                .groupby(GROUP, sort=False)["_d"]
                .agg(lambda s: ", ".join(sorted(set(s)))))
        for key in left[:SHOWN]:
            print(f"    {key[0]} {key[1]} -> {key[2]}: {what.get(key, '')}")
        if len(left) > SHOWN:
            print(f"    and {len(left) - SHOWN:,} more")
    p = presets(df, now)
    print("\n  The most downloaded packages' all-clear releases, the newest "
          "one each\n  (a preset for the site's analysed_clean case):")
    for _, x in p.iterrows():
        r = "" if x["_rank"] >= 10**9 else f"#{int(x['_rank'])}"
        day = ("date unknown" if pd.isna(x["_when"])
               else f"released {x['_when']:%Y-%m-%d}")
        print(f"    {r:>6}  {x['package']} {x['version_from']} -> "
              f"{x['version_to']}   {day}   "
              f"{plural(int(x['n']), 'row')}: {x['what']}")
    if p.empty:
        print("    none")

    print("\n5. WHAT THE LOADER STOPS WRITING (Varad's option (b))")
    print(f"  F1 and F39 rows in this file: {int(new.sum()):,} (the old "
          f"rules' {int(old.sum()):,}, {int(back.sum()):,} fewer)")
    print("  The deletion step counts again in the database, which also "
          "holds rows\n  from earlier loads, and deletes only the number "
          "its own dry run printed.")

    print("\n6. THE HOLDOUT, counts only")
    hb = int((back & held).sum())
    print(f"  {plural(hb, 'row')} come back into it, and "
          f"{plural(len((was - now) & in_held), 'upgrade')} that F1 or F39 "
          "had emptied")
    h = df[held]
    before, after = h[~old[held]], h[~new[held]]
    print(f"  pairs {before[GROUP].drop_duplicates().shape[0]:,} -> "
          f"{after[GROUP].drop_duplicates().shape[0]:,}, fingerprint "
          f"{fingerprint(before)} -> {fingerprint(after)}")
    print("  " + track(after, write=False)[0].strip())
    print(f"  gates: {MIN_POSITIVES}+ positives, {MIN_RANKABLE_PAIRS}+ pairs "
          "rankable at 10 and at 20.\n  Each label reads positives / "
          "rankable at 10 / rankable at 20:")
    for lab in [lab for lab in LABELS if lab in df]:
        was_g, now_g = gates(before, lab), gates(after, lab)
        ok = (now_g[0] >= MIN_POSITIVES and now_g[1] >= MIN_RANKABLE_PAIRS
              and now_g[2] >= MIN_RANKABLE_PAIRS)
        print(f"    {lab:<13} now {'/'.join(map(str, was_g))}, after "
              f"{'/'.join(map(str, now_g))}   "
              + ("clears them" if ok else "BELOW A GATE"))

    print("\n  This script changes nothing. The rules are version_strings() "
          "and\n  metadata_strings() in ml/features/build.py; the next build "
          "takes these rows\n  out there.")


if __name__ == "__main__":
    main()
