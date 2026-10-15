"""
Does the repair step set what §33.7 says, and nothing else, and only with
the count it was told?

    python scripts/test_db_repair.py

No database, no network, a few seconds. It builds a release table of
seventeen releases the way scripts/db_repair_counts.py reads it from
Postgres, runs the real planning and applying code against a stand-in
connection that records every statement, and checks each.

Five cases (NOTES §33.7):

  1. The rule, written out again: an `analysed` release's n_changes
     becomes its rows (rule 1), except one the loader counts (a
     version_to of changes.csv) with MORE rows than that count, which is
     db_prune's, listed and left; a version_from-only release, and a
     counted one with fewer rows than its count, are recounted; an `analysed`
     release with no rows that today's files do not name becomes unknown
     (rule 2); one today's files name is left to the loader and listed; a
     release of another status with rows is a hold, listed and never
     changed (rule 3); a release of another status with no rows and a
     stale count is set to 0.
  2. "Named by today's files" is the broad reading: changes.csv's
     version_to and version_from, and any releases.csv row, whatever its
     status; without releases.csv, changes.csv alone.
  3. The one transaction: the counts, then the status; the status update
     is guarded (still `analysed`, still no rows); nothing when there is
     nothing to do.
  4. --expect: none is a dry run that opens no transaction; the wrong
     number is refused before one opens; the right number applies and
     reads the table again; a consistent table says there is nothing to
     change; a database without migration 007 is refused before anything.
  5. The text carries the counts, the samples and the holds with their
     rows.
"""

import contextlib
import io
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scripts.db_repair_counts as rep  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


# ------------------------------------------------------------------ fixture

def table() -> pd.DataFrame:
    """One row per release: id, package, version, status, n_changes, rows.

     1  pkg 1.1   analysed, 3, 3 rows     consistent, nothing to do
     2  old 1.2   analysed, 0, 4 rows     not named, stale count: rule 1,
                                          0 -> 4
     3  old 1.3   analysed, 5, 2 rows     not named, stale count: rule 1,
                                          5 -> 2
     4  old 0.1   analysed, 0, 0 rows     not named: rule 2, unknown
     5  old 0.2   analysed, 2, 0 rows     not named: rules 1 and 2 both
     6  pkg 1.0   analysed, 0, 0 rows     named (version_from): the
                                          loader's, left alone, listed
     7  pkg 0.9   analysed_clean, 0, 2    rows under a clean verdict: hold
     8  old 0.3   no_baseline, 0, 1       rows under no_baseline: hold
     9  old 0.4   analysed_clean, 3, 0    stale count, no rows: rule 1 -> 0
    10  pkg 1.4   analysed_clean, 0, 0    consistent
    11  old 0.5   unknown, 0, 0           consistent
    12  old 0.6   analysed, 0, 0          named by releases.csv only: left
    13  old 0.7   analysed, 2, 3 rows     not named, has rows: rule 1 only,
                                          never rule 2
    14  pkg 1.3b  analysed, 2, 3 rows     counted (version_to), more rows than
                                          the loader's count: db_prune's
                                          superseded rows, listed, left
    15  pkg 0.9   analysed, 5, 2 rows     named as a version_from only: the
                                          loader writes it no count, so
                                          rule 1 recounts it
    16  pkg 1.2   analysed, 4, 3 rows     counted, FEWER rows than its
                                          count: rows deleted since the
                                          load, recounted
    17  old 0.8   analysed, 1, 3 rows     named by releases.csv only, MORE
                                          rows than its count: not the
                                          loader's count, so recounted
    """
    rows = [
        (1, "pkg", "1.1", "analysed", 3, 3),
        (2, "old", "1.2", "analysed", 0, 4),
        (3, "old", "1.3", "analysed", 5, 2),
        (4, "old", "0.1", "analysed", 0, 0),
        (5, "old", "0.2", "analysed", 2, 0),
        (6, "pkg", "1.0", "analysed", 0, 0),
        (7, "pkg", "0.9", "analysed_clean", 0, 2),
        (8, "old", "0.3", "no_baseline", 0, 1),
        (9, "old", "0.4", "analysed_clean", 3, 0),
        (10, "pkg", "1.4", "analysed_clean", 0, 0),
        (11, "old", "0.5", "unknown", 0, 0),
        (12, "old", "0.6", "analysed", 0, 0),
        (13, "old", "0.7", "analysed", 2, 3),
        (14, "pkg", "1.3b", "analysed", 2, 3),
        (15, "pkg", "0.9", "analysed", 5, 2),
        (16, "pkg", "1.2", "analysed", 4, 3),
        (17, "old", "0.8", "analysed", 1, 3),
    ]
    return pd.DataFrame(rows, columns=rep.COLUMNS)


def changes() -> pd.DataFrame:
    return pd.DataFrame({"package": ["pkg"] * 5,
                         "version_from": ["0.9", "1.0", "1.1", "1.2", "1.3"],
                         "version_to": ["1.0", "1.1", "1.2", "1.3", "1.3b"]})


def releases() -> pd.DataFrame:
    return pd.DataFrame({"package": ["pkg", "old", "old"],
                         "version": ["1.4", "0.6", "0.8"],
                         "status": ["analysed_clean", "pre_release",
                                    "analysed"]})


# -------------------------------------------------------------------- cases

def case_rule(p: dict) -> None:
    print("\n1. THE RULE, RELEASE BY RELEASE")
    check("rule 1: the `analysed` releases whose count is not their rows "
          "are recounted, and a stale count on an empty clean release goes "
          "to 0",
          sorted(p["recount"]),
          [(2, 0, 4), (3, 5, 2), (5, 2, 0), (9, 3, 0), (13, 2, 3),
           (15, 5, 2), (16, 4, 3), (17, 1, 3)])
    check("but not one the loader counts with MORE rows than its count: "
          "the extra rows are db_prune's, it is listed and left",
          (p["pruners"], 14 in [i for i, _o, _n in p["recount"]]),
          ([("pkg", "1.3b", 2, 3)], False))
    check("a version_from-only release gets no count from the loader, so "
          "it is recounted; a counted one with FEWER rows than its count "
          "too, and one named only by releases.csv with MORE rows than its "
          "count: the loader never counted it",
          [(i, o, n) for i, o, n in p["recount"] if i in (15, 16, 17)],
          [(15, 5, 2), (16, 4, 3), (17, 1, 3)])
    check("and the dry run says how many came from 0",
          p["recount_from_zero"], 1)
    check("rule 2: `analysed`, no rows, not named by today's files: unknown; "
          "one with rows is never relabelled, named or not",
          sorted(p["relabel"]), [4, 5])
    check("a release in both rules counts once", p["changed"], 9)
    check("an `analysed` release with no rows that today's files name is "
          "the loader's: left alone and listed",
          p["loaders"], [("pkg", "1.0"), ("old", "0.6")])
    check("rule 3: another status but rows in the table is a hold, with its "
          "count and rows",
          p["holds"], [(7, "pkg", "0.9", "analysed_clean", 0, 2),
                       (8, "old", "0.3", "no_baseline", 0, 1)])
    check("a consistent release of any status is not touched",
          {1, 10, 11} & (set(i for i, _o, _n in p["recount"])
                         | set(p["relabel"]) | set(h[0] for h in p["holds"])),
          set())


def case_named() -> None:
    print("\n2. WHAT TODAY'S FILES NAME")
    named = rep.named_today(changes(), releases())
    check("version_to, version_from, and every releases.csv row, whatever "
          "its status",
          named, {("pkg", "0.9"), ("pkg", "1.0"), ("pkg", "1.1"),
                  ("pkg", "1.2"), ("pkg", "1.3"), ("pkg", "1.3b"),
                  ("pkg", "1.4"), ("old", "0.6"), ("old", "0.8")})
    check("without releases.csv, changes.csv alone",
          rep.named_today(changes(), None),
          {("pkg", "0.9"), ("pkg", "1.0"), ("pkg", "1.1"), ("pkg", "1.2"),
           ("pkg", "1.3"), ("pkg", "1.3b")})
    check("the loader counts only the version_to releases",
          rep.counted_today(changes()),
          {("pkg", "1.0"), ("pkg", "1.1"), ("pkg", "1.2"), ("pkg", "1.3"),
           ("pkg", "1.3b")})
    p = rep.plan(table(), rep.named_today(changes(), None),
                 rep.counted_today(changes()))
    check("then old 0.6 is nobody's and becomes unknown too",
          sorted(p["relabel"]), [4, 5, 12])


class Conn:
    def __init__(self, rows, constraint=True):
        self.rows, self.constraint, self.sql = rows, constraint, []

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.sql.append((sql, params))
        if "pg_get_constraintdef" in sql:
            return _Result([("CHECK ((analysis_status = ANY (ARRAY["
                             "'analysed'::text, 'unknown'::text])))"
                             if self.constraint else
                             "CHECK ((analysis_status = ANY (ARRAY["
                             "'analysed'::text])))",)])
        if sql.startswith("SELECT r.id, p.name"):
            return _Result([tuple(r) for r in
                            self.rows.itertuples(index=False)])
        if sql.startswith("SELECT release_id, symbol_path"):
            return _Result([(7, "pkg.a", "OBJECT_REMOVED"),
                            (7, "pkg.b", "OBJECT_REMOVED"),
                            (8, "old.c", "PARAMETER_REMOVED")])
        if sql.startswith("UPDATE"):
            key = "ids"
            return _Result([], rowcount=len(params[key]))
        return _Result([])


class _Result:
    def __init__(self, rows, rowcount=0):
        self.rows, self.rowcount = rows, rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def scalar(self):
        return self.rows[0][0] if self.rows else None


def case_transaction(p: dict) -> None:
    print("\n3. THE ONE TRANSACTION, IN ORDER")
    conn = Conn(table())
    done = rep.apply(conn, p)
    heads = [s.split(" WHERE id")[0] for s, _ in conn.sql]
    check("the counts first, then the status",
          heads, ["UPDATE release SET n_changes = (SELECT count(*) FROM "
                  "breakage b WHERE b.release_id = release.id)",
                  "UPDATE release SET analysis_status = :unknown"])
    check("the recount names rule 1's releases, the relabel rule 2's",
          (conn.sql[0][1]["ids"], conn.sql[1][1]["ids"],
           conn.sql[1][1]["unknown"]),
          ([2, 3, 5, 9, 13, 15, 16, 17], [4, 5], "unknown"))
    check("the relabel is guarded: still `analysed`, still no rows",
          ("analysis_status = 'analysed'" in conn.sql[1][0],
           "NOT EXISTS (SELECT 1 FROM breakage b WHERE b.release_id = "
           "release.id)" in conn.sql[1][0]), (True, True))
    check("and reports what each step did",
          done, {"recounted": 8, "relabelled": 2})
    conn = Conn(table())
    done = rep.apply(conn, {"recount": [], "relabel": []})
    check("nothing to do, nothing sent", (conn.sql, done),
          ([], {"recounted": 0, "relabelled": 0}))


class Engine:
    def __init__(self, rows, constraint=True):
        self.conn = Conn(rows, constraint)
        self.began = 0

    @contextlib.contextmanager
    def connect(self):
        yield self.conn

    @contextlib.contextmanager
    def begin(self):
        self.began += 1
        yield self.conn


def _main(argv: list, rows: pd.DataFrame, constraint=True):
    engine = Engine(rows, constraint)
    saved = (rep.connect, rep.load_frames, rep.load_releases, sys.argv)
    rep.connect = lambda: engine
    rep.load_frames = lambda: (changes(), None, None)
    rep.load_releases = lambda: releases()
    sys.argv = ["db_repair_counts.py"] + argv
    out, code = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(out):
            rep.main()
    except SystemExit as e:
        code = str(e.code)
    finally:
        rep.connect, rep.load_frames, rep.load_releases, sys.argv = saved
    writes = [s for s, _ in engine.conn.sql if s.startswith("UPDATE")]
    return out.getvalue(), code, engine.began, writes


def case_expect() -> None:
    print("\n4. --expect: THE NUMBER READ IS THE NUMBER THAT CHANGES")
    out, code, began, writes = _main([], table())
    check("no --expect: a dry run that opens no transaction and writes "
          "nothing", (code, began, writes), (None, 0, []))
    check("and it prints the command with the count",
          "python scripts/db_repair_counts.py --expect 9" in out, True)
    out, code, began, writes = _main(["--expect", "4"], table())
    check("the wrong count is refused before any transaction opens",
          (code is not None and "REFUSING" in code and "has 9 to change"
           in code, began, writes), (True, 0, []))
    out, code, began, writes = _main(["--expect", "9"], table())
    check("the right count applies, in one transaction, and reads the "
          "table again",
          (code, began, len(writes), "would still change:" in out),
          (None, 1, 2, True))
    fine = table()[table()["id"].isin([1, 10, 11])]
    out, code, began, writes = _main([], fine)
    check("a consistent table: nothing to change, nothing opened",
          ("Nothing for rules 1 and 2 to change" in out, began, writes),
          (True, 0, []))
    holds_only = table()[table()["id"].isin([1, 7, 14])]
    out, code, began, writes = _main([], holds_only)
    check("a table with only a hold and a db_prune release: nothing for "
          "the rules, both listed",
          ("Nothing for rules 1 and 2 to change. The 1 rule 3 release(s) "
           "above are decided by hand." in out,
           "pkg 1.3b: count 2, rows 3" in out, began), (True, True, 0))
    out, code, began, writes = _main(["--expect", "9"], table(), False)
    check("a database without migration 007 is refused before anything",
          (code is not None and "migration 007" in code, began, writes),
          (True, 0, []))


def case_text(p: dict) -> None:
    print("\n5. THE TEXT")
    out = rep.describe(p, {7: [("pkg.a", "OBJECT_REMOVED"),
                              ("pkg.b", "OBJECT_REMOVED")],
                          8: [("old.c", "PARAMETER_REMOVED")]})
    check("the counts, by rule",
          ("n_changes to be set to the rows in the table: 8   (1 of them "
           "from 0)" in out, "to unknown: 2" in out,
           "left to the loader (where the files agree): 2" in out,
           "than its count: 1" in out,
           "      pkg 1.3b: count 2, rows 3" in out,
           "not touched here: 2" in out,
           "releases this would change: 9" in out),
          (True, True, True, True, True, True, True))
    check("the recounts and relabels by name",
          ("old 1.2: 0 -> 4" in out, "old 0.2: 2 -> 0" in out,
           "      old 0.1\n" in out), (True, True, True))
    check("the holds with their rows",
          ("pkg 0.9: analysed_clean, n_changes 0, 2 row(s) in the table"
           in out, "          pkg.a   OBJECT_REMOVED" in out,
           "old 0.3: no_baseline, n_changes 0, 1 row(s) in the table"
           in out), (True, True, True))


def main() -> None:
    p = rep.plan(table(), rep.named_today(changes(), releases()),
                 rep.counted_today(changes()))
    case_rule(p)
    case_named()
    case_transaction(p)
    case_expect()
    case_text(p)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDo not run scripts/db_repair_counts.py against Neon until "
              "this passes.")
        sys.exit(1)
    print("All checks passed. The repair sets counts from the table and")
    print("unknown on releases nobody compared, only with the count it was "
          "told.")


if __name__ == "__main__":
    main()
