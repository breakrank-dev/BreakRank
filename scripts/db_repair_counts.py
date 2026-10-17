"""
Repair what earlier loads left behind in the release table: n_changes that
is not the rows the table holds, and `analysed` on releases nobody
compared. Nothing else.

    python scripts/db_repair_counts.py                 # dry run: SELECTs only
    python scripts/db_repair_counts.py --expect 321    # apply, if that is the
                                                       # number of releases it
                                                       # would change now

THE RULE (NOTES §33.7, written before this code). A release's n_changes
and analysis_status were written by the load that saw it, and nothing
since read them against the rows the table actually holds: the loader
upserts and never removes (§8), db_prune deleted rows and recounted
nothing, and before 25 Sep a release the loader could not classify took
the column's default, `analysed` with 0 changes. Today's files cannot fix
it, because the loader writes only the releases they name. So:

  1. n_changes is the rows in the table, for every `analysed` release,
     whatever load wrote it, with one exception: a release the loader
     COUNTS (a version_to of today's changes.csv; the loader writes a
     count for no other release) keeps the loader's count when the table
     holds MORE rows for it than that. The extra rows are earlier loads'
     and this run did not produce them, which is db_prune's "superseded"
     and its job to remove, with its own recount; such releases are
     listed here and left. Fewer rows than the loader's count can only
     mean rows deleted since the load, and is recounted. A release of any
     other status with no rows and a count that is not 0 is set to 0 as
     well: a count of rows that are not there.
  2. An `analysed` release with no rows that today's files do not name
     becomes `unknown`: "diffed, changes found" is false for a release
     with none, and nothing has established it since the default of the
     day wrote it. A release today's files do name is the loader's to
     write, and is left alone and listed.
  3. A release of any other status that HAS rows (analysed_clean,
     no_baseline, no_source, yanked, analysis_failed, unknown) is not
     repaired by rule: a verdict of nothing found beside rows that say
     something was found, and which is stale cannot be read from the
     table. Each is listed in full, rows included, and decided by hand,
     written up in NOTES before anything is done about it. Since 18 Oct
     (NOTES §33.9) the listing says which of two other scripts' rules
     resolves a hold, where one does: a release today's files say was
     re-analysed (rule A, ml.db.live_releases) has rows db_prune removes
     as superseded; a release today's releases.csv calls not compared
     (rule B, ml.db.NOT_COMPARED, read off the file through the loader's
     own shield_verdicts) is written `analysed` with its rows' count by
     the next load. The rest stay decided by hand, and the note says why
     (not in today's files, or yanked today, or the table's analysed_clean
     kept over its rows). This step still changes none of them.
  4. Nothing else moves: no row is deleted, no status other than
     `analysed` is changed.

"Today's files name" a release (rule 2) when data/changes.csv has it as
a version_to or a version_from, or data/releases.csv has a row for it,
whatever its status: the broadest reading, so that the step leaves alone
everything the loader could have an opinion on. "The loader counts" a
release (rule 1) only when changes.csv has it as a version_to: the
loader writes n_changes for no other release.

--expect, as in scripts/db_drop_versions.py: the number read in the dry
run must be the number that changes, so this applies only when told the
count of releases it would change at that moment, in one transaction,
and refuses otherwise.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import (UNKNOWN, connect, live_releases,  # noqa: E402
                   load_frames, load_releases, package_rows,
                   planned_statuses, shield_verdicts)

COLUMNS = ["id", "package", "version", "status", "n_changes", "rows"]
SHOW = 12


def resolves(key: tuple, status: str, n_changes: int, rows: int,
             live: set, incoming: dict) -> str:
    """Which rule of NOTES §33.9 settles a rule 3 hold, in words, or that
    none does. A release today's files say was re-analysed holds rows an
    earlier run made, db_prune's superseded (rule A); one today's
    releases.csv calls not compared is written `analysed` with its rows'
    count by the next load (rule B). Rule A first: a re-analysed release's
    rows go, whatever its status reads. Rule B is read off TODAY'S FILE,
    not the table's status (the review's finding, 18 Oct), and through
    the loader's own shield_verdicts, so this says what the next load
    will do and nothing else; a release the loader will not write (not in
    today's files, named only as the older side of a pair, in a package
    the files disagree on, or in a package neither packages.csv nor
    changes.csv has, which main() filters out as write_all does) is
    nobody's."""
    if key in live:
        return ("rule A: today's files re-analysed it, so db_prune removes "
                "these rows as superseded")
    if key not in incoming:
        return ("neither rule: today's files do not name it, or only as the "
                "older side of a pair, or its package disagrees or is not "
                "loaded; decided by hand")
    row = {"package": key[0], "version": key[1],
           "analysis_status": incoming[key], "n_changes": 0}
    out, shielded, _clean = shield_verdicts(
        [row], {key: (status, n_changes, rows)})
    if shielded and out[0]["analysis_status"] == "analysed":
        return ("rule B: the next load writes `analysed` with these rows "
                "as its count")
    if shielded:
        return (f"neither rule: today's files call it {incoming[key]}, and "
                "rule B keeps the table's analysed_clean over its rows; "
                "decided by hand")
    return (f"neither rule: today's files call it {incoming[key]}; decided "
            "by hand")


def named_today(changes: pd.DataFrame, releases) -> set:
    """(package, version) for every release today's files name."""
    named = set(zip(changes["package"].astype(str),
                    changes["version_to"].astype(str)))
    named |= set(zip(changes["package"].astype(str),
                     changes["version_from"].astype(str)))
    if releases is not None and len(releases):
        named |= set(zip(releases["package"].astype(str),
                         releases["version"].astype(str)))
    return named


def counted_today(changes: pd.DataFrame) -> set:
    """(package, version) for every release the loader writes a count for:
    the version_to releases of changes.csv (ml/db.py, release_plan)."""
    return set(zip(changes["package"].astype(str),
                   changes["version_to"].astype(str)))


def plan(rows: pd.DataFrame, named: set, counted: set,
         live: set = frozenset(), incoming: dict | None = None) -> dict:
    """What the repair would do, release by release. Pure, so the test can
    check it; main() prints it and, with --expect, carries it out.

    rows: one row per release, COLUMNS, `rows` the count in the breakage
    table. `live`: the releases today's files say were re-analysed
    (ml.db.live_releases); `incoming`: the status today's files establish
    for each release (ml.db.planned_statuses); both read only to say which
    rule of §33.9 settles each hold. Returns the recounts (id, old, new),
    the relabels (id), the holds of rule 3 with what settles each, and the
    `analysed` releases with no rows that today's files name (left alone,
    listed)."""
    incoming = incoming or {}
    r = rows.copy()
    r["n_changes"] = r["n_changes"].fillna(0).astype(int)
    r["rows"] = r["rows"].fillna(0).astype(int)
    keys = [(str(p), str(v)) for p, v in zip(r["package"], r["version"])]
    r["named"] = [k in named for k in keys]
    r["counted"] = [k in counted for k in keys]
    analysed = r["status"].eq("analysed")
    off = r["n_changes"] != r["rows"]
    # The loader's count stands where the table merely holds MORE rows
    # than it: those rows are db_prune's. Everything else off is recounted.
    theirs = r["counted"] & (r["rows"] > r["n_changes"])
    recount = r[(analysed & off & ~theirs)
                | (~analysed & (r["rows"] == 0) & (r["n_changes"] != 0))]
    pruners = r[analysed & theirs]
    relabel = r[analysed & (r["rows"] == 0) & ~r["named"]]
    loaders = r[analysed & (r["rows"] == 0) & r["named"]]
    holds = r[~analysed & (r["rows"] > 0)]
    changed = set(recount["id"]) | set(relabel["id"])
    return {
        "releases": int(len(r)),
        "recount": [(int(i), int(o), int(n)) for i, o, n in
                    zip(recount["id"], recount["n_changes"], recount["rows"])],
        "recount_from_zero": int((recount["n_changes"] == 0).sum()),
        "relabel": [int(i) for i in relabel["id"]],
        "holds": [(int(i), str(p), str(v), str(s), int(n), int(k),
                   resolves((str(p), str(v)), str(s), int(n), int(k), live,
                            incoming))
                  for i, p, v, s, n, k in zip(
                      holds["id"], holds["package"], holds["version"],
                      holds["status"], holds["n_changes"], holds["rows"])],
        "loaders": [(str(p), str(v)) for p, v in
                    zip(loaders["package"], loaders["version"])],
        "pruners": [(str(p), str(v), int(o), int(n)) for p, v, o, n in
                    zip(pruners["package"], pruners["version"],
                        pruners["n_changes"], pruners["rows"])],
        "changed": len(changed),
        "sample_relabel": [(str(p), str(v)) for p, v in
                           zip(relabel["package"], relabel["version"])][:SHOW],
        "sample_recount": [(str(p), str(v), int(o), int(n)) for p, v, o, n in
                           zip(recount["package"], recount["version"],
                               recount["n_changes"], recount["rows"])][:SHOW],
    }


def describe(p: dict, hold_rows: dict) -> str:
    lines = [f"\nreleases in the table: {p['releases']:,}",
             f"  rule 1, n_changes to be set to the rows in the table: "
             f"{len(p['recount']):,}   ({p['recount_from_zero']:,} of them "
             "from 0)"]
    for name, ver, old, new in p["sample_recount"]:
        lines.append(f"      {name} {ver}: {old} -> {new}")
    if len(p["recount"]) > SHOW:
        lines.append(f"      ... and {len(p['recount']) - SHOW:,} more")
    lines.append(f"  rule 2, 'analysed' with no rows, not named by today's "
                 f"files, to unknown: {len(p['relabel']):,}")
    for name, ver in p["sample_relabel"]:
        lines.append(f"      {name} {ver}")
    if len(p["relabel"]) > SHOW:
        lines.append(f"      ... and {len(p['relabel']) - SHOW:,} more")
    lines.append(f"  'analysed' with no rows that today's files name, left "
                 f"to the loader (where the files agree): "
                 f"{len(p['loaders']):,}"
                 + ("" if not p["loaders"] else "   "
                    + ", ".join(f"{a} {b}" for a, b in p["loaders"][:SHOW])))
    lines.append(f"  'analysed', counted by the loader (a version_to of "
                 f"today's changes.csv), more rows\n  than its count: "
                 f"{len(p['pruners']):,}   (the extra rows are db_prune's "
                 "superseded rows; it recounts when\n      it removes them; "
                 "left here)")
    for name, ver, old, new in p["pruners"][:SHOW]:
        lines.append(f"      {name} {ver}: count {old}, rows {new}")
    if len(p["pruners"]) > SHOW:
        lines.append(f"      ... and {len(p['pruners']) - SHOW:,} more")
    lines.append(f"  rule 3, another status but rows in the table, not "
                 f"touched here: {len(p['holds']):,}"
                 + ("" if not p["holds"] else
                    "\n      (each with the rule of NOTES §33.9 that settles "
                    "it, or none)"))
    for i, name, ver, status, n, k, how in p["holds"]:
        lines.append(f"      {name} {ver}: {status}, n_changes {n}, "
                     f"{k} row(s) in the table\n          {how}")
        for sym, kind in hold_rows.get(i, [])[:SHOW]:
            lines.append(f"          {sym}   {kind}")
        if len(hold_rows.get(i, [])) > SHOW:
            lines.append(f"          ... and {len(hold_rows[i]) - SHOW:,} "
                         "more")
    lines.append(f"\nreleases this would change: {p['changed']:,}")
    return "\n".join(lines)


def apply(conn, p: dict) -> dict:
    """The one transaction: the counts, then the status. Returns what each
    step did."""
    from sqlalchemy import text

    done = {"recounted": 0, "relabelled": 0}
    ids = [i for i, _o, _n in p["recount"]]
    if ids:
        done["recounted"] = conn.execute(text("""
            UPDATE release SET n_changes = (SELECT count(*) FROM breakage b
                                            WHERE b.release_id = release.id)
            WHERE id = ANY(:ids)"""), {"ids": ids}).rowcount
    if p["relabel"]:
        done["relabelled"] = conn.execute(text("""
            UPDATE release SET analysis_status = :unknown
            WHERE id = ANY(:ids) AND analysis_status = 'analysed'
              AND NOT EXISTS (SELECT 1 FROM breakage b
                              WHERE b.release_id = release.id)"""),
            {"ids": p["relabel"], "unknown": UNKNOWN}).rowcount
    return done


def fetch(conn) -> pd.DataFrame:
    """One row per release with its count of rows in the breakage table."""
    from sqlalchemy import text

    found = conn.execute(text("""
        SELECT r.id, p.name, r.version, r.analysis_status, r.n_changes,
               (SELECT count(*) FROM breakage b WHERE b.release_id = r.id)
        FROM release r
        JOIN package p ON p.id = r.package_id
        ORDER BY p.name, r.version
    """)).fetchall()
    return pd.DataFrame([tuple(r) for r in found], columns=COLUMNS)


def rows_of(conn, ids: list) -> dict:
    """The rows of the rule 3 releases, for the listing."""
    from sqlalchemy import text

    if not ids:
        return {}
    found = conn.execute(text("""
        SELECT release_id, symbol_path, kind FROM breakage
        WHERE release_id = ANY(:ids) ORDER BY release_id, symbol_path
    """), {"ids": ids}).fetchall()
    out: dict = {}
    for rid, sym, kind in found:
        out.setdefault(int(rid), []).append((str(sym), str(kind)))
    return out


def check_statuses(conn) -> None:
    """The CHECK constraint must name `unknown` (migration 007), or rule 2
    would abort the transaction on its first row."""
    from sqlalchemy import text

    found = conn.execute(text("""
        SELECT pg_get_constraintdef(oid) FROM pg_constraint
        WHERE conrelid = 'release'::regclass
          AND pg_get_constraintdef(oid) LIKE '%analysis_status%'""")).scalar()
    if not found or f"'{UNKNOWN}'" not in found:
        sys.exit("This database does not accept the status 'unknown': "
                 "migration 007 is not applied.\nNothing was changed.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Set n_changes from the table and `unknown` on "
                    "releases nobody compared (NOTES §33.7). Dry run by "
                    "default.")
    ap.add_argument("--expect", type=int, default=None,
                    help="apply, if and only if this is the number of "
                         "releases the dry run says it would change. "
                         "Without it nothing is written.")
    args = ap.parse_args()

    changes, _usage, packages = load_frames()
    releases = load_releases()
    named = named_today(changes, releases)
    counted = counted_today(changes)
    live = live_releases(changes, releases)
    # The loader writes releases only for the packages it writes (write_all,
    # package_rows: packages.csv and changes.csv), so the note reads the
    # same set, or it would promise a load for a package nobody loads.
    loaded = {p["name"] for p in package_rows(changes, packages)}
    incoming = {k: v for k, v in planned_statuses(changes, releases).items()
                if k[0] in loaded}
    engine = connect()
    with engine.connect() as conn:
        check_statuses(conn)
        p = plan(fetch(conn), named, counted, live, incoming)
        hold_rows = rows_of(conn, [h[0] for h in p["holds"]])
    print(describe(p, hold_rows))

    n = p["changed"]
    if not n:
        print("\nNothing for rules 1 and 2 to change." + (
                  "" if not p["holds"] else
                  f" The {len(p['holds'])} rule 3 release(s) above are "
                  "settled as each says, by\ndb_prune, by the next load, or "
                  "by hand (NOTES §33.9)."))
        return
    if args.expect is None:
        print(f"\nDry run. Nothing was changed. To apply rules 1 and 2 to "
              f"exactly these {n:,} releases:\n  python "
              f"scripts/db_repair_counts.py --expect {n}\nFirst: send this "
              "output to Varad, with the rule 3 decisions proposed (NOTES "
              "§33.7, §33.9).")
        return
    if args.expect != n:
        sys.stdout.flush()
        sys.exit(f"\nREFUSING: you expected {args.expect:,} releases and the "
                 f"table has {n:,} to change now.\nNothing was changed. Read "
                 f"the dry run above; if it is right, re-run with --expect "
                 f"{n}.")

    with engine.begin() as conn:
        done = apply(conn, p)
    print(f"\nrecounted n_changes on {done['recounted']:,} releases and set "
          f"{done['relabelled']:,} to unknown, in one transaction.")
    with engine.connect() as conn:
        after = plan(fetch(conn), named, counted, live, incoming)
    print(f"releases the rules would still change: {after['changed']:,} "
          f"(should be 0); rule 3 holds: {len(after['holds']):,}.")
    print("Next: the read-only checks (scripts/db_model_check.py, "
          "scripts/db_extras.py, the status count), then db_prune's dry run; "
          "after its delete, this dry run once more.")


if __name__ == "__main__":
    main()
