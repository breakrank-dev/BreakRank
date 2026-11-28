"""
Does the repair step set what §33.7 says, and nothing else, and only with
the count it was told?

    python scripts/test_db_repair.py

No database, no network, a few seconds. It builds a release table of
twenty-three releases the way scripts/db_repair_counts.py reads it from
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
    No (package, version) twice: the real table has them unique.

     1  pkg 1.1   analysed, 3, 3 rows     consistent, nothing to do
     2  old 1.2   analysed, 0, 4 rows     not named, stale count: rule 1,
                                          0 -> 4
     3  old 1.3   analysed, 5, 2 rows     not named, stale count: rule 1,
                                          5 -> 2
     4  old 0.1   analysed, 0, 0 rows     not named: rule 2, unknown
     5  old 0.2   analysed, 2, 0 rows     not named: rules 1 and 2 both
     6  pkg 1.0   analysed, 0, 0 rows     named (version_from): the
                                          loader's, left alone, listed
     7  pkg 1.4   analysed_clean, 0, 2    rows under a clean verdict, and
                                          today's agreeing releases.csv
                                          calls it analysed_clean: hold,
                                          re-analysed, rule A (db_prune's)
     8  old 0.3   no_baseline, 0, 1       rows under no_baseline, in no
                                          file of today's: hold, neither
                                          rule (the loader never writes it)
     9  old 0.4   analysed_clean, 3, 0    stale count, no rows: rule 1 -> 0
    10  pkg 1.5   analysed_clean, 0, 0    consistent
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
    18  old 0.95  analysed_clean, 0, 1    rows under a clean verdict, in no
                                          file of today's: hold, neither
    19  old 0.97  yanked, 0, 1            rows under yanked, in no file of
                                          today's: hold, neither
    20  pkg 1.6   no_baseline, 0, 1       the table says not compared, but
                                          today's agreeing releases.csv says
                                          analysed_clean: re-analysed, rule
                                          A (db_prune's), not rule B
    21  pkg 0.8   unknown, 0, 1           rows under unknown, and today's
                                          releases.csv calls it no_baseline:
                                          hold, rule B (the next load writes
                                          analysed, 1)
    22  pkg 0.7   no_baseline, 0, 1       rows under no_baseline, and today's
                                          releases.csv calls it yanked: hold,
                                          neither (yanked is written as it
                                          comes)
    23  pkg 0.6   analysed_clean, 0, 1    rows under a clean verdict, and
                                          today's releases.csv calls it
                                          no_baseline: hold, neither (rule B
                                          keeps the table's analysed_clean)
    """
    rows = [
        (1, "pkg", "1.1", "analysed", 3, 3),
        (2, "old", "1.2", "analysed", 0, 4),
        (3, "old", "1.3", "analysed", 5, 2),
        (4, "old", "0.1", "analysed", 0, 0),
        (5, "old", "0.2", "analysed", 2, 0),
        (6, "pkg", "1.0", "analysed", 0, 0),
        (7, "pkg", "1.4", "analysed_clean", 0, 2),
        (8, "old", "0.3", "no_baseline", 0, 1),
        (9, "old", "0.4", "analysed_clean", 3, 0),
        (10, "pkg", "1.5", "analysed_clean", 0, 0),
        (11, "old", "0.5", "unknown", 0, 0),
        (12, "old", "0.6", "analysed", 0, 0),
        (13, "old", "0.7", "analysed", 2, 3),
        (14, "pkg", "1.3b", "analysed", 2, 3),
        (15, "pkg", "0.9", "analysed", 5, 2),
        (16, "pkg", "1.2", "analysed", 4, 3),
        (17, "old", "0.8", "analysed", 1, 3),
        (18, "old", "0.95", "analysed_clean", 0, 1),
        (19, "old", "0.97", "yanked", 0, 1),
        (20, "pkg", "1.6", "no_baseline", 0, 1),
        (21, "pkg", "0.8", "unknown", 0, 1),
        (22, "pkg", "0.7", "no_baseline", 0, 1),
        (23, "pkg", "0.6", "analysed_clean", 0, 1),
    ]
    return pd.DataFrame(rows, columns=rep.COLUMNS)


PAIRS = [("0.9", "1.0"), ("1.0", "1.1"), ("1.1", "1.2"), ("1.2", "1.3"),
         ("1.3", "1.3b")]


def changes() -> pd.DataFrame:
    """pkg's five pairs, one real change each, with the columns the
    loader's release_plan reads (live_releases goes through it)."""
    return pd.DataFrame([
        {"package": "pkg", "package_rank": 1, "version_from": vf,
         "version_to": vt, "symbol": f"pkg.f_{vt}", "kind": "OBJECT_REMOVED",
         "sub_target": "", "released_at": "2026-01-01",
         "explanation": f"x.py:1: pkg.f_{vt}: Public object was removed"}
        for vf, vt in PAIRS])


def releases() -> pd.DataFrame:
    """pkg agrees with changes.csv (its five analysed releases, one change
    each), so its analysed_clean rows count as re-analysed: 1.4, 1.5 and
    1.6; its 0.8 is no_baseline today, its 0.7 yanked, its 0.6 no_baseline.
    old is not in changes.csv, so releases.csv's old 0.8 `analysed` makes
    the two files disagree on it, and nothing of old's is used but the
    names."""
    rows = [("pkg", "0.6", "no_baseline", None),
            ("pkg", "0.7", "yanked", None),
            ("pkg", "0.8", "no_baseline", None),
            ("pkg", "1.4", "analysed_clean", 0),
            ("pkg", "1.5", "analysed_clean", 0),
            ("pkg", "1.6", "analysed_clean", 0),
            ("old", "0.6", "pre_release", None),
            ("old", "0.8", "analysed", 2)]
    rows += [("pkg", vt, "analysed", 1) for _vf, vt in PAIRS]
    return pd.DataFrame([{"package": p, "version": v, "released_at": "",
                          "status": s, "n_changes": n}
                         for p, v, s, n in rows])


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
          [h[:6] for h in p["holds"]],
          [(7, "pkg", "1.4", "analysed_clean", 0, 2),
           (8, "old", "0.3", "no_baseline", 0, 1),
           (18, "old", "0.95", "analysed_clean", 0, 1),
           (19, "old", "0.97", "yanked", 0, 1),
           (20, "pkg", "1.6", "no_baseline", 0, 1),
           (21, "pkg", "0.8", "unknown", 0, 1),
           (22, "pkg", "0.7", "no_baseline", 0, 1),
           (23, "pkg", "0.6", "analysed_clean", 0, 1)])
    check("and each hold says which rule of §33.9 settles it: re-analysed "
          "today, rule A, whatever the table's status; else today's file "
          "calls it not compared and the table has rows, rule B; else "
          "neither",
          [h[6].split(":")[0] for h in p["holds"]],
          ["rule A", "neither rule", "neither rule", "neither rule",
           "rule A", "rule B", "neither rule", "neither rule"])
    check("rule B is read off today's file, not the table's status: old 0.3 "
          "(no_baseline in the table, in no file) is nobody's, pkg 0.8 "
          "(unknown in the table, no_baseline in the file) is rule B's, and "
          "the note says why the rest are nobody's",
          [h[6] for h in p["holds"] if h[0] in (8, 21, 22, 23)],
          ["neither rule: today's files do not name it, or only as the older "
           "side of a pair, or its package disagrees or is not loaded; "
           "decided by hand",
           "rule B: the next load writes `analysed` with these rows as its "
           "count",
           "neither rule: today's files call it yanked; decided by hand",
           "neither rule: today's files call it no_baseline, and rule B "
           "keeps the table's analysed_clean over its rows; decided by hand"])
    check("a consistent release of any status is not touched",
          {1, 10, 11} & (set(i for i, _o, _n in p["recount"])
                         | set(p["relabel"]) | set(h[0] for h in p["holds"])),
          set())
    check("a hold is never recounted or relabelled, whatever settles it",
          {7, 8, 18, 19, 20, 21, 22, 23}
          & (set(i for i, _o, _n in p["recount"]) | set(p["relabel"])), set())


def case_named() -> None:
    print("\n2. WHAT TODAY'S FILES NAME")
    named = rep.named_today(changes(), releases())
    check("version_to, version_from, and every releases.csv row, whatever "
          "its status",
          named, {("pkg", "0.9"), ("pkg", "1.0"), ("pkg", "1.1"),
                  ("pkg", "1.2"), ("pkg", "1.3"), ("pkg", "1.3b"),
                  ("pkg", "1.4"), ("pkg", "1.5"), ("pkg", "1.6"),
                  ("pkg", "0.6"), ("pkg", "0.7"), ("pkg", "0.8"),
                  ("old", "0.6"), ("old", "0.8")})
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
    live = rep.live_releases(changes(), releases())
    check("re-analysed today (ml.db.live_releases): the version_to releases "
          "and the analysed_clean ones of an agreeing releases.csv; not old "
          "0.8, whose package disagrees, and not a version_from alone",
          live, {("pkg", "1.0"), ("pkg", "1.1"), ("pkg", "1.2"),
                 ("pkg", "1.3"), ("pkg", "1.3b"), ("pkg", "1.4"),
                 ("pkg", "1.5"), ("pkg", "1.6")})
    check("the statuses today's files establish, as the loader plans them: "
          "the version_to releases analysed, releases.csv's rows of the "
          "agreeing package, nothing of old's",
          rep.planned_statuses(changes(), releases()),
          {("pkg", "1.0"): "analysed", ("pkg", "1.1"): "analysed",
           ("pkg", "1.2"): "analysed", ("pkg", "1.3"): "analysed",
           ("pkg", "1.3b"): "analysed", ("pkg", "1.4"): "analysed_clean",
           ("pkg", "1.5"): "analysed_clean", ("pkg", "1.6"): "analysed_clean",
           ("pkg", "0.6"): "no_baseline", ("pkg", "0.7"): "yanked",
           ("pkg", "0.8"): "no_baseline"})
    check("without releases.csv, the version_to releases alone",
          rep.live_releases(changes(), None),
          {("pkg", "1.0"), ("pkg", "1.1"), ("pkg", "1.2"), ("pkg", "1.3"),
           ("pkg", "1.3b")})
    p = rep.plan(table(), rep.named_today(changes(), releases()),
                 rep.counted_today(changes()),
                 rep.live_releases(changes(), None),
                 rep.planned_statuses(changes(), None))
    check("without releases.csv no hold is re-analysed or named by a file, "
          "so every one is nobody's",
          sorted(set(h[6].split(":")[0] for h in p["holds"])),
          ["neither rule"])
    k = ("a", "1")
    check("the words of each, and rule A before rule B",
          (rep.resolves(k, "no_baseline", 0, 2, {k}, {k: "analysed_clean"}),
           rep.resolves(k, "yanked", 0, 2, set(), {k: "no_source"}),
           rep.resolves(k, "analysed", 0, 2, set(), {k: "analysis_failed"}),
           rep.resolves(k, "no_baseline", 0, 2, set(), {})),
          ("rule A: today's files re-analysed it, so db_prune removes these "
           "rows as superseded",
           "rule B: the next load writes `analysed` with these rows as its "
           "count",
           "rule B: the next load writes `analysed` with these rows as its "
           "count",
           "neither rule: today's files do not name it, or only as the older "
           "side of a pair, or its package disagrees or is not loaded; "
           "decided by hand"))


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


def _main(argv: list, rows: pd.DataFrame, constraint=True, packages=None,
          rels=None):
    engine = Engine(rows, constraint)
    saved = (rep.connect, rep.load_frames, rep.load_releases, sys.argv)
    rep.connect = lambda: engine
    rep.load_frames = lambda: (changes(), None, packages)
    rep.load_releases = lambda: releases() if rels is None else rels
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
           "above are settled as each says, by\ndb_prune, by the next load, "
           "or by hand (NOTES §33.9)." in out,
           "pkg 1.3b: count 2, rows 3" in out, began), (True, True, 0))
    out, code, began, writes = _main([], table())
    check("the dry run reads the live set from both files and says what "
          "settles each hold",
          ("pkg 1.4: analysed_clean, n_changes 0, 2 row(s) in the table\n"
           "          rule A: today's files re-analysed it" in out,
           "pkg 0.8: unknown, n_changes 0, 1 row(s) in the table\n"
           "          rule B: the next load writes `analysed`" in out,
           "old 0.3: no_baseline, n_changes 0, 1 row(s) in the table\n"
           "          neither rule: today's files do not name it" in out,
           "pkg 0.7: no_baseline, n_changes 0, 1 row(s) in the table\n"
           "          neither rule: today's files call it yanked" in out),
          (True, True, True, True))
    # a package in releases.csv alone: the loader writes its releases only
    # if packages.csv lists it (write_all's pkg_id), so the note follows
    solo = pd.DataFrame([{"package": "solo", "version": "1.0",
                          "released_at": "", "status": "no_baseline",
                          "n_changes": None}])
    rows = pd.concat([table(), pd.DataFrame(
        [(24, "solo", "1.0", "unknown", 0, 1)], columns=rep.COLUMNS)],
        ignore_index=True)
    rels = pd.concat([releases(), solo], ignore_index=True)
    listed = pd.DataFrame({"package": ["pkg", "solo"], "download_rank": [1, 2],
                           "github_repo": [None, None]})
    out_without, _c, _b, _w = _main([], rows, rels=rels)
    out_with, _c, _b, _w = _main([], rows, packages=listed, rels=rels)
    check("a hold in a package only releases.csv names is nobody's unless "
          "packages.csv lists the package, as for the loader (the review's "
          "finding)",
          ("solo 1.0: unknown, n_changes 0, 1 row(s) in the table\n"
           "          neither rule: today's files do not name it, or only as "
           "the older side of a pair, or its package disagrees or is not "
           "loaded" in out_without,
           "solo 1.0: unknown, n_changes 0, 1 row(s) in the table\n"
           "          rule B: the next load writes `analysed`" in out_with),
          (True, True))
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
           "not touched here: 8\n      (each with the rule of NOTES §33.9 "
           "that settles it, or none)" in out,
           "releases this would change: 9" in out),
          (True, True, True, True, True, True, True))
    check("the recounts and relabels by name",
          ("old 1.2: 0 -> 4" in out, "old 0.2: 2 -> 0" in out,
           "      old 0.1\n" in out), (True, True, True))
    check("the holds with their rows",
          ("pkg 1.4: analysed_clean, n_changes 0, 2 row(s) in the table"
           in out, "          pkg.a   OBJECT_REMOVED" in out,
           "old 0.3: no_baseline, n_changes 0, 1 row(s) in the table"
           in out), (True, True, True))


def main() -> None:
    p = rep.plan(table(), rep.named_today(changes(), releases()),
                 rep.counted_today(changes()),
                 rep.live_releases(changes(), releases()),
                 rep.planned_statuses(changes(), releases()))
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
