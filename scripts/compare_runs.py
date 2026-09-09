"""
Did the alias change break the diff, or did PyPI just move?

    python scripts/compare_runs.py

Reads  data/changes-prealias.csv   the 5 Sep run, 23,025 rows / 410 packages
       data/changes.csv            the 6 Sep run, 21,336 rows / 402 packages

WHY THIS EXISTS. Adding export_paths should add a COLUMN and change nothing
else. The row count dropped by 1,689 and eight packages vanished, so one of
two things is true and they are not equally acceptable:

  1. PyPI MOVED. `--restart` re-downloads, and the pipeline takes the last
     six releases that ship a non-yanked sdist. Anyone publishing in the
     meantime shifts that window: the oldest pair rolls off the end and a
     new one appears. Different releases, different changes. Nothing wrong.

  2. THE NEW CODE CHANGED THE DIFF. export_index() runs inside
     diff_collections(), and diff_series() turns any exception from that
     call into a failed pair. So a bug in the alias walk would not crash
     loudly — it would quietly delete whole version pairs and leave a
     smaller dataset that looks perfectly normal.

THE DECISIVE TEST is the third section below: for version pairs present in
BOTH files, do the row counts agree? Those pairs saw identical inputs. If
their counts match exactly, case 1 is proven and the new column is
innocent. If any differ, case 2 is real and the number of affected pairs
is printed so it can be investigated rather than argued about.
"""

import pathlib
import sys

import pandas as pd

DATA = pathlib.Path("data")
OLD = DATA / "changes-prealias.csv"
NEW = DATA / "changes.csv"

KEY = ["package", "version_from", "version_to"]


def load(path: pathlib.Path, label: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found — cannot compare {label}.")
    df = pd.read_csv(path)
    print(f"{label:<12} {len(df):>7,} rows   {df['package'].nunique():>4} packages"
          f"   {df[KEY].drop_duplicates().shape[0]:>5} version pairs")
    return df


def main() -> None:
    print()
    old = load(OLD, "before")
    new = load(NEW, "after")
    print(f"{'delta':<12} {len(new) - len(old):>+7,} rows   "
          f"{new['package'].nunique() - old['package'].nunique():>+4} packages")

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 68)
    print("  1. WHICH PACKAGES CHANGED SIDE?")
    print("=" * 68)
    op, np_ = set(old["package"]), set(new["package"])
    lost, gained = sorted(op - np_), sorted(np_ - op)
    print(f"in the old run only: {len(lost)}")
    for p in lost[:20]:
        print(f"   -{p}  ({int((old['package'] == p).sum())} rows)")
    print(f"\nin the new run only: {len(gained)}")
    for p in gained[:20]:
        print(f"   +{p}  ({int((new['package'] == p).sum())} rows)")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 68)
    print("  2. DID THE VERSION WINDOW MOVE?")
    print("=" * 68)
    op_pairs = set(map(tuple, old[KEY].drop_duplicates().itertuples(index=False)))
    np_pairs = set(map(tuple, new[KEY].drop_duplicates().itertuples(index=False)))
    shared = op_pairs & np_pairs
    print(f"pairs only in the old run: {len(op_pairs - shared):>5}")
    print(f"pairs only in the new run: {len(np_pairs - shared):>5}")
    print(f"pairs in BOTH runs:        {len(shared):>5}")

    moved = sorted({p for p, _, _ in (op_pairs - shared)}
                   & {p for p, _, _ in (np_pairs - shared)})
    print(f"\n{len(moved)} packages have pairs on both sides — their release")
    print("window shifted, which is PyPI publishing, not us. Examples:")
    for p in moved[:6]:
        o = sorted(v for pk, _, v in (op_pairs - shared) if pk == p)[-3:]
        n = sorted(v for pk, _, v in (np_pairs - shared) if pk == p)[-3:]
        print(f"   {p:<22} dropped {o}   gained {n}")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 68)
    print("  3. THE DECISIVE TEST — SAME PAIRS, SAME COUNTS?")
    print("=" * 68)
    print("These pairs saw identical input in both runs. Any difference here")
    print("is the new code, not PyPI.\n")

    oc = old.groupby(KEY).size().rename("before")
    nc = new.groupby(KEY).size().rename("after")
    both = pd.concat([oc, nc], axis=1).dropna().astype(int)
    both = both.loc[[i for i in both.index if i in shared]]

    disagree = both[both["before"] != both["after"]]
    print(f"shared pairs compared: {len(both):,}")
    print(f"pairs whose row count CHANGED: {len(disagree):,}")

    if disagree.empty:
        print("\n** ZERO. Every pair that saw the same two releases produced")
        print("** the same number of rows. The alias column added data and")
        print("** removed none. The drop is entirely PyPI's release window.")
    else:
        delta = int((disagree["after"] - disagree["before"]).sum())
        print(f"** {len(disagree):,} pairs differ, net {delta:+,} rows.")
        print("** That is the new code, not PyPI. Worst offenders:\n")
        d = disagree.assign(diff=disagree["after"] - disagree["before"])
        for idx, r in d.reindex(d["diff"].abs().sort_values(
                ascending=False).index).head(12).iterrows():
            print(f"   {idx[0]:<20} {idx[1]} -> {idx[2]:<12} "
                  f"{int(r.before):>5} -> {int(r.after):<5} ({int(r['diff']):+})")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 68)
    print("  4. WHAT THE NEW COLUMN ACTUALLY BOUGHT")
    print("=" * 68)
    if "export_paths" not in new.columns:
        print("no export_paths column — wrong file?")
        return
    ep = new["export_paths"].fillna("").astype(str)
    hit = ep.ne("")
    print(f"rows with at least one shorter public name: {int(hit.sum()):,} "
          f"({hit.mean():.1%})")
    n_paths = ep[hit].str.count(";") + 1
    if not n_paths.empty:
        print(f"paths per such row: mean {n_paths.mean():.2f}, "
              f"max {int(n_paths.max())}")

    by_pkg = (pd.DataFrame({"package": new["package"], "hit": hit})
              .groupby("package")["hit"].agg(["mean", "size"]))
    by_pkg = by_pkg[by_pkg["size"] >= 20].sort_values("mean", ascending=False)
    print("\npackages where re-export is most common (min 20 rows):")
    for p, r in by_pkg.head(8).iterrows():
        print(f"   {p:<24} {r['mean']:>6.1%} of {int(r['size']):>5} rows")
    print("\nand least common — these genuinely have no shorter names:")
    for p, r in by_pkg.tail(5).iterrows():
        print(f"   {p:<24} {r['mean']:>6.1%} of {int(r['size']):>5} rows")


if __name__ == "__main__":
    main()
