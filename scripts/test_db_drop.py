"""
Does the deletion step remove exactly the version-string and
package-metadata rows, and nothing without the count it was told?

    python scripts/test_db_drop.py

No database, no network, a few seconds. It builds a breakage table of
fourteen rows the way scripts/db_drop_versions.py reads it from Postgres,
runs the real marking, planning and deleting code against a stand-in
connection that records every statement, and checks each; then
scripts/db_prune.py's classification, which must name those rows as a
category of its own and never delete them.

Nine cases (NOTES §33.5, §33.6, §33.7):

  1. The rule, written out again: a version constant or a metadata string
     whose VALUE changed goes; a removed one, one changed to `unset`, a
     class's VERSION and a row the rule cannot read stay. build.py's own
     two functions decide, from symbol_path, kind and griffe_message.
  2. Varad's breakrank-fixture rows are counted and left alone.
  3. The plan: which releases lose rows, which are left with none, and
     which of those read 'analysed' and so become analysed_clean.
  4. The one transaction, in order: the rows' predictions, the rows, the
     releases' recount, the clean status; without migration 005 no release
     is touched.
  5. --expect: no number means a dry run that opens no transaction; the
     wrong number is refused before one opens; the right number deletes
     and reads the table again afterwards.
  6. db_prune.py puts these rows in their own category first, whatever
     else they are, and its deletions are what they were.
  7. db_extras.py counts them on a line of their own too, reading the
     griffe message out of `detail` however the driver hands it back, and
     builds "what we sent" from the rows the loader sends.
  8. db_drop_split.py (read only) splits the marked rows and the emptied
     releases into today's changes.csv and earlier loads, and compares the
     file's all-clear with the table's both ways (Varad's question of
     15 Oct).
  9. db_prune.py recounts the releases it deletes rows from, in the same
     transaction, and names a release left with no row instead of
     relabelling it (NOTES §33.7); its classification reads rows that
     carry the release_id as an eighth field.
"""

import contextlib
import io
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scripts.db_drop_split as split_  # noqa: E402
import scripts.db_drop_versions as drop  # noqa: E402
import scripts.db_extras as extras  # noqa: E402
import scripts.db_prune as prune  # noqa: E402
from ml.db import FIXTURE_PACKAGE  # noqa: E402

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

CHANGED, REMOVED = "ATTRIBUTE_CHANGED_VALUE", "OBJECT_REMOVED"


def _msg(title, old, new):
    return f"{title}: Attribute value was changed: {old} -> {new}"


def table() -> pd.DataFrame:
    """The breakage table as fetch() returns it: id, package, version,
    symbol, kind, explanation (detail.griffe_message, the file and line
    already stripped by the loader), release_id, status, n_changes.

    release 1  pkg 1.1  three rows, two of them version strings: left with
               one, stays analysed
    release 2  pkg 1.2  one row, its version string: emptied, analysed ->
               analysed_clean
    release 3  pkg 1.3  rows that look like the rule but are not: all stay
    release 4  old 0.5  one metadata row, from an earlier load, status
               no_baseline (aged out, never a verdict): emptied, not clean
    release 5  the fixture package: its version string is left alone
    """
    rows = [
        (1, "pkg", "1.1", "pkg.__version__", CHANGED,
         _msg("__version__", "'1.0'", "'1.1'"), 1, "analysed", 3),
        (2, "pkg", "1.1", "pkg._version.version", CHANGED,
         _msg("version", "'1.0'", "'1.1'"), 1, "analysed", 3),
        (3, "pkg", "1.1", "pkg.core.run", REMOVED,
         "run: Public object was removed", 1, "analysed", 3),
        (4, "pkg", "1.2", "pkg.__version__", CHANGED,
         _msg("__version__", "'1.1'", "'1.2'"), 2, "analysed", 1),
        (5, "pkg", "1.3", "pkg.__version__", REMOVED,
         "__version__: Public object was removed", 3, "analysed", 5),
        (6, "pkg", "1.3", "pkg._version.version", CHANGED,
         _msg("version", "'1.2'", "unset"), 3, "analysed", 5),
        (7, "pkg", "1.3", "pkg.core.Config.VERSION", CHANGED,
         _msg("Config.VERSION", "1", "2"), 3, "analysed", 5),
        (8, "pkg", "1.3", "pkg.core.VERSION", CHANGED, None, 3,
         "analysed", 5),
        (9, "pkg", "1.3", "pkg.__about__.__author__", REMOVED,
         "__author__: Public object was removed", 3, "analysed", 5),
        (10, "old", "0.5", "old.__about__.__copyright__", CHANGED,
         _msg("__copyright__", "'2024'", "'2025'"), 4, "no_baseline", 0),
        (11, FIXTURE_PACKAGE, "9.9", "fixture.__version__", CHANGED,
         _msg("__version__", "'9.8'", "'9.9'"), 5, "analysed", 2),
        (12, FIXTURE_PACKAGE, "9.9", "fixture.thing", REMOVED,
         "thing: Public object was removed", 5, "analysed", 2),
        (13, "pkg", "1.3", "pkg.__VERSION__", CHANGED, None, 3,
         "analysed", 5),
        (14, "pkg", "1.3", "pkg.__about__.__title__", CHANGED,
         _msg("__title__", "'pkg'", "'PKG'"), 3, "analysed", 5),
    ]
    return pd.DataFrame(rows, columns=drop.COLUMNS)


# -------------------------------------------------------------------- cases

def case_rule(m: pd.DataFrame) -> None:
    print("\n1. THE RULE: A CHANGED VALUE GOES, EVERYTHING ELSE STAYS")
    rule = dict(zip(m["id"], m["rule"]))
    check("a changed __version__ and a module's changed `version` are "
          "version strings",
          (rule[1], rule[2], rule[4]),
          ("version string", "version string", "version string"))
    check("a removed __version__, one changed to unset, and a class's "
          "VERSION stay",
          (rule[5], rule[6], rule[7]), ("", "", ""))
    check("a bare VERSION with no explanation stays: when in doubt the row "
          "stays", rule[8], "")
    check("a dunder with no explanation goes: the name alone decides for "
          "those (build.py's rule)", rule[13], "version string")
    check("a changed copyright or title is package metadata; a removed "
          "__author__ is a change",
          (rule[10], rule[14], rule[9]),
          ("package metadata", "package metadata", ""))


def case_fixture(p: dict) -> None:
    print("\n2. VARAD'S FIXTURE ROWS ARE COUNTED AND LEFT ALONE")
    check("the fixture's version string is not among the doomed",
          11 in p["doomed"], False)
    check("but it is counted", p["fixture_skipped"], 1)
    check("the doomed rows are exactly the six changed values outside it",
          sorted(p["doomed"]), [1, 2, 4, 10, 13, 14])
    check("by rule: 4 version strings, 2 metadata", p["by_rule"],
          {"version string": 4, "package metadata": 2})
    check("by name, most first",
          list(p["by_leaf"].items())[0], ("__version__", 2))


def case_plan(p: dict) -> None:
    print("\n3. THE PLAN: RELEASES THAT LOSE ROWS, AND THE ONES LEFT EMPTY")
    check("four releases lose rows", sorted(p["releases"]), [1, 2, 3, 4])
    check("what each is left with: 1 keeps one row, 2 and 4 none, 3 keeps "
          "five", p["left"], {1: 1, 2: 0, 3: 5, 4: 0})
    check("the emptied ones", sorted(p["emptied"]), [2, 4])
    check("only the emptied release reading 'analysed' becomes "
          "analysed_clean; no_baseline is not a verdict to overwrite",
          p["to_clean"], [2])
    out = drop.describe(p, [("lambdarank-label_alias", 3)])
    check("the dry run's text carries the counts",
          ("not breakrank-fixture's: 6  (42.9% of the table)" in out,
           "releases that lose rows: 4; left with none: 2, of which 1 read "
           "'analysed'" in out,
           "lambdarank-label_alias 3" in out), (True, True, True))


class Conn:
    """Records every statement and its parameters; answers the reads the
    script makes with the fixture table."""

    def __init__(self, rows: pd.DataFrame, has_status=True):
        self.rows, self.has_status, self.sql = rows, has_status, []

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.sql.append((sql, params))
        if "information_schema.columns" in sql:
            return _Result([(1,)] if self.has_status else [])
        if sql.startswith("SELECT b.id, p.name"):
            return _Result([tuple(r) for r in
                            self.rows.itertuples(index=False)])
        if "SELECT model_version, count(*)" in sql:
            return _Result([("lambdarank-label_alias", 3)])
        if sql.startswith(("DELETE", "UPDATE")):
            n = len(params["ids"]) if "ids" in params else len(params["rids"])
            return _Result([], rowcount=n)
        return _Result([])


class _Result:
    def __init__(self, rows, rowcount=0):
        self.rows, self.rowcount = rows, rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


def case_transaction(p: dict) -> None:
    print("\n4. THE ONE TRANSACTION, IN ORDER")
    conn = Conn(table())
    done = drop.delete(conn, p, has_status=True)
    kinds = [s.split(" WHERE breakage_id")[0].split(" WHERE id")[0]
             for s, _ in conn.sql]
    check("predictions first, then the rows, then the recount, then the "
          "clean status",
          kinds, ["DELETE FROM prediction", "DELETE FROM breakage",
                  "UPDATE release SET n_changes = (SELECT count(*) FROM "
                  "breakage b WHERE b.release_id = release.id)",
                  "UPDATE release SET analysis_status = 'analysed_clean'"])
    check("the deletes name exactly the doomed ids",
          [params["ids"] for _, params in conn.sql[:2]],
          [p["doomed"], p["doomed"]])
    check("the recount names the releases that lost rows, and the clean "
          "update is guarded by n_changes = 0 AND analysis_status = "
          "'analysed'",
          (conn.sql[2][1]["rids"], "n_changes = 0 AND analysis_status = "
           "'analysed'" in conn.sql[3][0]), (p["releases"], True))
    check("and reports what each step did",
          done, {"prediction": 6, "breakage": 6, "recounted": 4, "clean": 4,
                 "left_marked": None})
    conn = Conn(table(), has_status=False)
    drop.delete(conn, p, has_status=False)
    check("without migration 005 no release is touched",
          [s.split(" ")[0] for s, _ in conn.sql], ["DELETE", "DELETE"])


class Engine:
    def __init__(self, rows, has_status=True):
        self.conn = Conn(rows, has_status)
        self.began = 0

    @contextlib.contextmanager
    def connect(self):
        yield self.conn

    @contextlib.contextmanager
    def begin(self):
        self.began += 1
        yield self.conn


def _main(argv: list, rows: pd.DataFrame):
    engine = Engine(rows)
    original_connect, original_argv = drop.connect, sys.argv
    drop.connect, sys.argv = (lambda: engine), ["db_drop_versions.py"] + argv
    out, code = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(out):
            drop.main()
    except SystemExit as e:
        code = str(e.code)
    finally:
        drop.connect, sys.argv = original_connect, original_argv
    writes = [s for s, _ in engine.conn.sql if s.startswith(("DELETE",
                                                             "UPDATE"))]
    return out.getvalue(), code, engine.began, writes


def case_expect() -> None:
    print("\n5. --expect: THE NUMBER READ IS THE NUMBER THAT GOES")
    out, code, began, writes = _main([], table())
    check("no --expect: a dry run that opens no transaction and writes "
          "nothing", (code, began, writes), (None, 0, []))
    check("and it prints the command with the count",
          "python scripts/db_drop_versions.py --expect 6" in out, True)
    out, code, began, writes = _main(["--expect", "5"], table())
    check("the wrong count is refused before any transaction opens",
          (code is not None and "REFUSING" in code and "table has 6 now"
           in code, began, writes), (True, 0, []))
    out, code, began, writes = _main(["--expect", "6"], table())
    check("the right count deletes, in one transaction",
          (code, began, len(writes)), (None, 1, 4))
    check("and the table is read again afterwards",
          "rows the rules mark, after:" in out, True)
    empty = table().iloc[0:0]
    out, code, began, writes = _main(["--expect", "0"], empty)
    check("an empty table: nothing to do, nothing opened",
          ("Nothing to do" in out, began), (True, 0))


def case_prune() -> None:
    print("\n6. db_prune NAMES THESE ROWS AS ITS OWN CATEGORY, FIRST")
    found = [
        # (id, package, version, symbol, kind, sub_target, griffe_message)
        (1, "pkg", "1.1", "pkg.__version__", CHANGED, "",
         _msg("__version__", "'1.0'", "'1.1'")),
        (2, "pkg", "1.1", "pkg.core.run", REMOVED, "",
         "run: Public object was removed"),
        (3, "pkg", "1.0", "pkg.__about__.__copyright__", CHANGED, "",
         _msg("__copyright__", "'2024'", "'2025'")),
        (4, "pkg", "1.1", "python.pkg.x", REMOVED, "", "x: removed"),
        (5, "pkg", "1.1", "pkg.gone", REMOVED, "", "gone: removed"),
        (6, "pkg", "0.9", "pkg.older", REMOVED, "", "older: removed"),
        (7, "pkg", "0.9", "pkg.__version__", CHANGED, "",
         _msg("__version__", "'0.8'", "'0.9'")),
        (8, FIXTURE_PACKAGE, "1.0.0", "fixture.__version__", CHANGED, "",
         _msg("__version__", "'0.9'", "'1.0.0'")),
    ]
    ours = {("pkg", "1.1", "pkg.__version__", CHANGED, ""),
            ("pkg", "1.1", "pkg.core.run", REMOVED, "")}
    live = {("pkg", "1.1")}
    groups = prune.classify(found, ours, live)
    check("a version string in changes.csv (1), one from an earlier load "
          "(7) and a metadata row (3) are this category, not superseded "
          "or aged", groups["versions"], [1, 3, 7])
    check("the other groups are what they were: bad root, superseded, "
          "aged out, the fixture's version string among the aged as before",
          (groups["bad"], groups["superseded"], groups["aged"]),
          ([4], [5], [6, 8]))
    check("a row in changes.csv that is a change is in no group",
          2 in sum(groups.values(), []), False)
    check("an empty table classifies to nothing",
          prune.classify([], ours, live),
          {"versions": [], "bad": [], "superseded": [], "aged": []})


def case_extras() -> None:
    print("\n7. db_extras COUNTS THEM ON A LINE OF THEIR OWN")
    import json

    k_vs = ("pkg", "1.1", "pkg.__version__", CHANGED, "")
    k_meta = ("pkg", "1.0", "pkg.__about__.__copyright__", CHANGED, "")
    k_run = ("pkg", "1.1", "pkg.core.run", REMOVED, "")
    k_unset = ("pkg", "1.2", "pkg._version.version", CHANGED, "")
    k_fix = (FIXTURE_PACKAGE, "1.0.0", "fixture.__version__", CHANGED, "")
    k_none = ("pkg", "1.3", "pkg.__version__", CHANGED, "")
    in_db = {
        # detail as a dict (psycopg's JSONB), as a string, and missing
        k_vs: (1, {"griffe_message": _msg("__version__", "'1.0'", "'1.1'")},
               False),
        k_meta: (2, json.dumps({"griffe_message":
                                _msg("__copyright__", "'a'", "'b'")}), False),
        k_run: (3, {"griffe_message": "run: Public object was removed"},
                False),
        k_unset: (4, {"griffe_message": _msg("version", "'1.1'", "unset")},
                  False),
        k_fix: (5, {"griffe_message": _msg("__version__", "'0.9'", "'1.0.0'")},
                False),
        k_none: (6, None, False),
    }
    extra = sorted(in_db)
    got = extras.not_sent_any_more(extra, in_db)
    check("a changed __version__ and a changed copyright are the rows the "
          "loader stopped writing, whether detail is a dict or a string; "
          "a dunder with no detail too", got, {k_vs, k_meta, k_none})
    check("a removal, a value changed to unset, and the fixture's row are not",
          {k_run, k_unset, k_fix} & got, set())
    check("the griffe message is read out of any shape of detail",
          (extras.griffe_message({"griffe_message": "x"}),
           extras.griffe_message('{"griffe_message": "y"}'),
           extras.griffe_message(None), extras.griffe_message("not json")),
          ("x", "y", "", ""))
    check("nothing to classify, nothing marked",
          extras.not_sent_any_more([], {}), set())


def _chg(pkg, vf, vt, sym, kind, title=None, old="'1'", new="'2'"):
    """A changes.csv row with griffe's explanation, as the ingest writes
    it (the title is the object's bare name for a module-level attribute)."""
    title = title or sym.rsplit(".", 1)[-1]
    msg = (f"Attribute value was changed: {old} -> {new}"
           if kind == CHANGED else "Public object was removed")
    return {"package": pkg, "version_from": vf, "version_to": vt,
            "symbol": sym, "kind": kind, "sub_target": "",
            "explanation": f"x/y.py:3: {title}: {msg}"}


def case_split() -> None:
    print("\n8. db_drop_split: TODAY'S ROWS AND EARLIER LOADS, TOLD APART")
    # today's changes.csv: pkg 1.1 (two version strings and a change),
    # pkg 1.2 (its version string only), pkg 1.3 (its rows, __VERSION__
    # among them, no __title__), pkg 1.4 (its version string only). Not
    # in it: old 0.5, and 1.3's __title__ row.
    changes = pd.DataFrame([
        _chg("pkg", "1.0", "1.1", "pkg.__version__", CHANGED),
        _chg("pkg", "1.0", "1.1", "pkg._version.version", CHANGED),
        _chg("pkg", "1.0", "1.1", "pkg.core.run", REMOVED),
        _chg("pkg", "1.1", "1.2", "pkg.__version__", CHANGED),
        _chg("pkg", "1.2", "1.3", "pkg.__version__", REMOVED),
        _chg("pkg", "1.2", "1.3", "pkg._version.version", CHANGED,
             new="unset"),
        _chg("pkg", "1.2", "1.3", "pkg.core.Config.VERSION", CHANGED,
             title="Config.VERSION"),
        _chg("pkg", "1.2", "1.3", "pkg.__VERSION__", CHANGED),
        _chg("pkg", "1.3", "1.4", "pkg.__version__", CHANGED),
    ])
    # the table: the fixture of the other cases, plus pkg 1.4 with its
    # version string and one row an earlier load left there, and a
    # copyright row an earlier load left in pkg 1.1
    rows = pd.concat([table(), pd.DataFrame([
        (15, "pkg", "1.4", "pkg.__version__", CHANGED,
         _msg("__version__", "'1.3'", "'1.4'"), 6, "analysed", 2),
        (16, "pkg", "1.4", "pkg.old_thing", REMOVED,
         "old_thing: Public object was removed", 6, "analysed", 2),
        (17, "pkg", "1.1", "pkg.__about__.__copyright__", CHANGED,
         _msg("__copyright__", "'2024'", "'2025'"), 1, "analysed", 3),
    ], columns=drop.COLUMNS)], ignore_index=True)
    s = split_.split(rows, changes)
    check("marked rows split: 5 of today's files (1, 2, 4, 13, 15), 3 "
          "from earlier loads",
          (s["marked"], s["today"], s["earlier"]), (8, 5, 3))
    check("of the earlier three, two sit in releases today's files still "
          "name (1.1's copyright, 1.3's __title__), one aged out (old 0.5)",
          (s["earlier_live"], s["earlier_aged"]), (2, 1))
    check("the file's own count, as §33.2 made it: 5 rows, 5 keys",
          (s["file_marked"], s["file_keys"]), (5, 5))
    check("releases left with nothing: pkg 1.2 (today's) and old 0.5 "
          "(earlier); pkg 1.4 keeps its earlier-load row",
          (s["emptied"], s["emptied_today"], s["emptied_earlier"]),
          (2, 1, 1))
    check("the file's all-clear has 1.2 and 1.4; the table empties only "
          "1.2, and the odd one is named",
          (s["file_clear"], s["both"], s["file_only"], s["table_only"]),
          (2, 1, [("pkg", "1.4")], []))
    out = split_.describe(s)
    check("the text carries the counts and the odd release",
          ("rows of today's changes.csv:              5" in out,
           "from earlier loads:                       3" in out,
           "file only: pkg 1.4" in out,
           "other rows" in out), (True, True, True, True))
    nothing = split_.split(table().iloc[0:0], changes)
    check("an empty table: nothing marked, nothing emptied, the file's "
          "all-clear still counted",
          (nothing["marked"], nothing["emptied"], nothing["file_clear"]),
          (0, 0, 2))


class PruneConn:
    """Records db_prune's statements; answers the recount's read-back with
    the releases left empty."""

    def __init__(self, emptied):
        self.sql, self.emptied = [], emptied

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.sql.append((sql, params))
        if sql.startswith("UPDATE"):
            return _Result([], rowcount=len(params["rids"]))
        if sql.startswith("SELECT p.name, r.version, r.analysis_status"):
            return _Result(self.emptied)
        return _Result([])


def case_prune_recount() -> None:
    print("\n9. db_prune RECOUNTS WHAT IT DELETES FROM, AND NAMES THE EMPTY")
    conn = PruneConn([("pkg", "1.1", "analysed")])
    n, emptied = prune.recount(conn, [3, 1, 2])
    check("one UPDATE sets n_changes from the table for the releases given, "
          "then the empty ones are read back by name",
          ([s.split(" WHERE")[0] for s, _ in conn.sql],
           conn.sql[0][1]["rids"], n, emptied),
          (["UPDATE release SET n_changes = (SELECT count(*) FROM breakage b",
            "SELECT p.name, r.version, r.analysis_status FROM release r JOIN "
            "package p ON p.id = r.package_id"],
           [3, 1, 2], 3, [("pkg", "1.1", "analysed")]))
    check("the read-back asks for n_changes = 0 among those releases only",
          "WHERE r.id = ANY(:rids) AND r.n_changes = 0" in conn.sql[1][0],
          True)
    check("no status is changed by it",
          any("analysis_status =" in s for s, _ in conn.sql), False)
    conn = PruneConn([])
    check("no release to recount: nothing sent",
          (prune.recount(conn, []), conn.sql), ((0, []), []))
    found = [
        (1, "pkg", "1.1", "python.pkg.x", REMOVED, "", "x: removed", 11),
        (2, "pkg", "1.1", "pkg.core.run", REMOVED, "",
         "run: Public object was removed", 11),
    ]
    groups = prune.classify(found, {("pkg", "1.1", "pkg.core.run", REMOVED,
                                     "")}, {("pkg", "1.1")})
    check("classify reads rows that carry release_id as an eighth field",
          (groups["bad"], groups["superseded"]), ([1], []))

    # The whole delete path of main(), against a stand-in engine: the two
    # DELETEs, then the recount, then the names of the emptied releases.
    class PruneEngine:
        def __init__(self):
            self.conn, self.began = PruneConn([("pkg", "1.1", "analysed")]), 0

        @contextlib.contextmanager
        def connect(self):
            yield self.conn

        @contextlib.contextmanager
        def begin(self):
            self.began += 1
            yield self.conn

    table_rows = [
        (1, "pkg", "1.1", "python.pkg.x", REMOVED, "", "x: removed", 11),
        (2, "pkg", "1.1", "python.pkg.y", REMOVED, "", "y: removed", 11),
        (3, "pkg", "1.2", "pkg.core.run", REMOVED, "",
         "run: Public object was removed", 12),
    ]

    class TableConn(PruneConn):
        has_count = True

        def execute(self, stmt, params=None):
            sql = " ".join(str(stmt).split())
            if "information_schema.columns" in sql:
                self.sql.append((sql, params))
                if "n_changes" in sql:
                    return _Result([(1,)] if self.has_count else [])
                return _Result([(1,)])
            if sql.startswith("SELECT b.id, p.name, r.version"):
                self.sql.append((sql, params))
                return _Result(table_rows)
            if sql.startswith("DELETE"):
                self.sql.append((sql, params))
                return _Result([], rowcount=len(params["ids"]))
            return super().execute(stmt, params)

    engine = PruneEngine()
    engine.conn = TableConn([("pkg", "1.1", "analysed")])
    changes = pd.DataFrame([{
        "package": "pkg", "package_rank": 1, "version_from": "1.1",
        "version_to": "1.2", "symbol": "pkg.core.run", "kind": REMOVED,
        "sub_target": "", "is_private": False, "in_dunder_all": False,
        "module_depth": 2, "is_top_level": False, "released_at": "2026-01-01",
        "inherited_by": 0, "was_deprecated_before": False,
        "explanation": "x.py:3: run: Public object was removed"}])
    def run_prune(engine):
        saved = (prune.connect, prune.load_frames, sys.argv)
        prune.connect = lambda: engine
        prune.load_frames = lambda: (changes, None, None)
        # two of three rows is over the 5% guard, which is not what this
        # case tests, so the guard is raised for it
        sys.argv = ["db_prune.py", "--yes", "--max-share", "1.0"]
        out, code = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(out):
                prune.main()
        except SystemExit as e:
            code = str(e.code)
        finally:
            prune.connect, prune.load_frames, sys.argv = saved
        return out, code

    out, code = run_prune(engine)
    check("the prune ran to the end", code, None)
    kinds = [s.split(" ")[0] for s, _ in engine.conn.sql if
             s.startswith(("DELETE", "UPDATE", "SELECT p.name"))]
    check("with --yes: the two deletes, the recount of the releases that "
          "lost rows, the read-back of the empty ones, in one transaction",
          (kinds, engine.began), (["DELETE", "DELETE", "UPDATE", "SELECT"],
                                  1))
    update = next(params for s, params in engine.conn.sql
                  if s.startswith("UPDATE"))
    check("the recount names the release the un-importable rows came from",
          update["rids"], [11])
    check("and the output names the release left empty, status unchanged, "
          "and says the repair step relabels it, not a person",
          ("n_changes set from the table on 1 release(s)" in out.getvalue(),
           "**   pkg 1.1   analysed" in out.getvalue(),
           "NOT changed here" in out.getvalue(),
           "becomes `unknown` on the next" in out.getvalue()
           and "db_repair_counts.py" in out.getvalue()),
          (True, True, True, True))
    engine = PruneEngine()
    engine.conn = TableConn([])
    engine.conn.has_count = False
    out, code = run_prune(engine)
    check("on a database without migration 005 the delete still runs, and "
          "nothing is recounted, said so",
          (code, [s.split(" ")[0] for s, _ in engine.conn.sql
                  if s.startswith(("DELETE", "UPDATE"))],
           "no n_changes column (migration 005 is not applied)"
           in out.getvalue()), (None, ["DELETE", "DELETE"], True))


def main() -> None:
    rows = table()
    m = drop.marked(rows)
    p = drop.plan(rows)
    case_rule(m)
    case_fixture(p)
    case_plan(p)
    case_transaction(p)
    case_expect()
    case_prune()
    case_extras()
    case_split()
    case_prune_recount()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDo not run scripts/db_drop_versions.py against Neon until "
              "this passes.")
        sys.exit(1)
    print("All checks passed. The deletion step removes the rows build.py's")
    print("rules mark, only with the count it was told, in one transaction.")


if __name__ == "__main__":
    main()
