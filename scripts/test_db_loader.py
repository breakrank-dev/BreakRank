"""
Does the loader write what it should to the database, on a RE-load too?

    python scripts/test_db_loader.py

No database, no network, a few seconds. It builds four tiny packages,
runs the real release_plan() and write_all() against a stand-in database
connection that records every SQL statement, and checks both.

WHY THIS FILE EXISTS. Measured on Neon on 25 Sep 2026:

    breakage rows with inherited_by > 0     18     (changes.csv: 647)
    release.analysis_status                 'analysed' on all 2,236 rows

Two loader bugs, both silent. A re-load only updated `detail`, so every
value computed after a row's first load (inherited_by above all) never
reached the database. And releases.csv (NOTES §14) was never loaded, so a
release that changed nothing either read "analysed, 0 changes" or was not
in the database at all, and the site answered "not tracked" for it.

And a third, found the same day: the API's sentences read `old_value`,
`new_value` and `was_deprecated_in` from `detail`, and the loader never
wrote any of them, so those sentences always fell back to "X changed in
this release."

Seven cases:

  1. A package where releases.csv agrees with changes.csv gets every
     status, including the clean newest release that used to be missing.
  2. A package where they DISAGREE gets none of releases.csv's statuses:
     one run's verdict is never stamped onto another run's data, and
     nothing unknown is ever written as analysed_clean.
  3. Statuses the database does not accept (pre_release, dev_release, and
     no_baseline before migration 006) are skipped, not forced.
  4. Without releases.csv, only releases that changed something get a
     status, with the count changes.csv gives.
  5. The SQL itself: a re-load refreshes every column it writes, and the
     release upsert carries the status only when the database has it.
  6. The sentence fields, from the real ingest differ: a changed default's
     two values (even a default that contains " -> "), and the version
     where a deprecation marker was seen.
  7. A package with changes but no row in packages.csv still gets written,
     with all its rows. The first dry run on the real data would have
     written 10,006 of 23,268 breakage rows, because packages.csv listed
     187 of the 314 packages.

The same fixture was also loaded into a real Postgres built from Varad's
migrations 001-006, 001-005 and 001-004, first with the old loader
(reproducing both bugs exactly as Neon shows them) and then with this
one. That needs a Postgres server, so it is not part of this file.
"""

import contextlib
import io
import pathlib
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ml.db as db  # noqa: E402

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

def _row(pkg, vf, vt, sym, when, inherited=0, private=False):
    return {"package": pkg, "package_rank": 1, "version_from": vf,
            "version_to": vt, "symbol": sym, "kind": "OBJECT_REMOVED",
            "sub_target": "", "is_private": private, "in_dunder_all": False,
            "module_depth": sym.count("."),
            "is_top_level": sym.count(".") == 1,
            "released_at": when, "inherited_by": inherited,
            "explanation": f"x.py:3: {sym}: Public object was removed",
            "was_deprecated_before": False}


def changes() -> pd.DataFrame:
    """alpha: releases.csv agrees.  beta: it disagrees.  gamma: it has no
    rows for it.  delta (below): changed nothing, releases.csv only."""
    return pd.DataFrame([
        _row("alpha", "1.0.0", "1.1.0", "alpha.f1", "2026-03-01"),
        _row("alpha", "1.0.0", "1.1.0", "alpha.f2", "2026-03-01"),
        _row("alpha", "1.0.0", "1.1.0", "alpha.core.C.m", "2026-03-01",
             private=True),
        _row("alpha", "1.2.0", "1.3.0", "alpha.base.Mixin.go", "2026-05-01",
             inherited=5),
        _row("alpha", "1.2.0", "1.3.0", "alpha.g", "2026-05-01"),
        _row("beta", "2.0.0", "2.1.0", "beta.a", "2026-04-01"),
        _row("beta", "2.0.0", "2.1.0", "beta.b", "2026-04-01"),
        _row("beta", "2.1.0", "2.2.0", "beta.c", "2026-06-01"),
        _row("gamma", "0.1", "0.2", "gamma.base.B.run", "2026-02-01",
             inherited=3),
    ])


RELEASES_CSV = """package,version,released_at,status,n_changes
alpha,1.4.1.dev1,,dev_release,
alpha,1.5.0rc1,,pre_release,
alpha,1.3.5,,yanked,
alpha,1.0.0,2026-02-01,no_baseline,
alpha,1.1.0,2026-03-01,analysed,3
alpha,1.2.0,2026-04-01,analysed_clean,0
alpha,1.3.0,2026-05-01,analysed,2
alpha,1.4.0,2026-06-01,analysed_clean,0
beta,2.0.0,2026-03-01,no_baseline,
beta,2.1.0,2026-04-01,analysed,2
beta,2.2.0,2026-06-01,analysed,4
delta,3.0.0,2026-01-01,no_baseline,
delta,3.1.0,2026-02-01,analysed_clean,0
delta,3.2.0,2026-03-01,analysed_clean,0
delta,3.2.0,2026-03-01,analysed_clean,0
"""


def releases() -> pd.DataFrame:
    """Through the real load_releases(), duplicate row and all."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-loader-"))
    path = tmp / "releases.csv"
    path.write_text(RELEASES_CSV)
    original = db.RELEASES
    db.RELEASES = path
    try:
        return db.load_releases()
    finally:
        db.RELEASES = original


def by_key(rows: list) -> dict:
    return {(r["package"], r["version"]): r for r in rows}


# -------------------------------------------------------------------- cases

def case_agree(known: dict, unknown: dict, report: dict) -> None:
    print("\n1. WHERE THE FILES AGREE, EVERY STATUS IS LOADED")
    check("alpha is the package the files agree on (and delta, which "
          "changed nothing)", report["agree"], ["alpha", "delta"])
    check("alpha's releases, oldest to newest",
          [(v, known[("alpha", v)]["analysis_status"],
            known[("alpha", v)]["n_changes"])
           for v in ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.3.5", "1.4.0")],
          [("1.0.0", "no_baseline", 0), ("1.1.0", "analysed", 3),
           ("1.2.0", "analysed_clean", 0), ("1.3.0", "analysed", 2),
           ("1.3.5", "yanked", 0), ("1.4.0", "analysed_clean", 0)])
    check("the clean NEWEST release is now written, with its date and bump",
          {k: known[("alpha", "1.4.0")][k]
           for k in ("released_at", "bump_type")},
          {"released_at": "2026-06-01", "bump_type": "minor"})
    check("a yanked release is not anyone's predecessor: 1.4.0's bump is "
          "measured from 1.3.0", known[("alpha", "1.4.0")]["bump_type"],
          "minor")
    check("delta, a package that changed nothing, gets its releases "
          "(the duplicate row is collapsed)",
          sorted((v, r["analysis_status"]) for (p, v), r in known.items()
                 if p == "delta"),
          [("3.0.0", "no_baseline"), ("3.1.0", "analysed_clean"),
           ("3.2.0", "analysed_clean")])


def case_disagree(known: dict, unknown: dict, report: dict) -> None:
    print("\n2. WHERE THEY DISAGREE, releases.csv IS NOT USED")
    check("beta is named, with what differs (only in changes, only in "
          "releases, recounted)", report["disagree"], {"beta": (0, 0, 1)})
    check("beta 2.2.0 keeps the count changes.csv gives, not releases.csv's",
          (known[("beta", "2.2.0")]["analysis_status"],
           known[("beta", "2.2.0")]["n_changes"]), ("analysed", 1))
    check("beta 2.0.0 and gamma 0.1 get NO status, not a guessed one",
          sorted(k for k in unknown), [("beta", "2.0.0"), ("gamma", "0.1")])
    check("gamma is listed as having no releases.csv rows at all",
          report["uncovered"], ["gamma"])
    check("nothing whose status is unknown is ever written as analysed_clean",
          [k for k, r in known.items() if r["analysis_status"] ==
           "analysed_clean" and k[0] not in report["agree"]], [])


def case_skipped(chg: pd.DataFrame, rel: pd.DataFrame) -> None:
    print("\n3. STATUSES THE DATABASE DOES NOT ACCEPT ARE SKIPPED")
    known, unknown, report = db.release_plan(chg, rel)
    check("pre_release and dev_release are skipped and counted",
          report["skipped"], {"dev_release": 1, "pre_release": 1})
    no_006 = db.DB_STATUSES - {"no_baseline"}
    known, unknown, report = db.release_plan(chg, rel, no_006)
    statuses = {r["analysis_status"] for r in known}
    # 2, not 3: beta's no_baseline row is never considered at all, because
    # releases.csv disagrees with changes.csv about beta (case 2).
    check("before migration 006, no_baseline is skipped too, never written",
          ("no_baseline" in statuses, report["skipped"].get("no_baseline")),
          (False, 2))


def case_no_releases(chg: pd.DataFrame) -> None:
    print("\n4. WITHOUT releases.csv, ONLY WHAT changes.csv PROVES")
    known, unknown, report = db.release_plan(chg, None)
    check("the releases that changed something are analysed, with counts",
          sorted((r["package"], r["version"], r["n_changes"]) for r in known),
          [("alpha", "1.1.0", 3), ("alpha", "1.3.0", 2), ("beta", "2.1.0", 2),
           ("beta", "2.2.0", 1), ("gamma", "0.2", 1)])
    check("and the report says nothing else is claimed",
          "Nothing else is claimed" in db.describe_plan(known, unknown,
                                                        report), True)


class FakeConn:
    """Records every statement write_all sends, and answers its two
    read-backs with ids for the rows it was given."""

    def __init__(self):
        self.sql, self.pkg, self.rel, self.brk = [], {}, [], []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.sql.append(sql)
        if "INSERT INTO package" in sql:
            self.pkg[params["name"]] = len(self.pkg) + 1
            return _Result([(self.pkg[params["name"]],)])
        if "INSERT INTO release" in sql:
            self.rel += params
        if "INSERT INTO breakage" in sql:
            self.brk += params
        if "SELECT r.id, p.name, r.version" in sql:
            names = {i: n for n, i in self.pkg.items()}
            return _Result([(i, names[r["package_id"]], r["version"])
                            for i, r in enumerate(self.rel, 1)])
        return _Result([])


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


def case_sql(chg: pd.DataFrame, rel: pd.DataFrame) -> None:
    print("\n5. THE SQL: A RE-LOAD REFRESHES EVERY COLUMN IT WRITES")
    caps = {"sub_target": True, "inherited_by": True, "positive_rate": True,
            "analysis_status": True, "statuses": set(db.DB_STATUSES)}
    usage = pd.DataFrame({"symbol": ["alpha.f1"], "user_count": [4]})
    conn = FakeConn()
    with contextlib.redirect_stdout(io.StringIO()):
        db.write_all(conn, chg, usage, None, caps, None, rel)
    brk = next(s for s in conn.sql if "INSERT INTO breakage" in s)
    update = brk.split("DO UPDATE", 1)[1]
    check("on conflict, a breakage row refreshes every non-key column",
          sorted(c for c in ("inherited_by", "is_private", "module_depth",
                             "is_top_level", "in_dunder_all", "detail")
                 if f"{c} = EXCLUDED.{c}" in update),
          ["detail", "in_dunder_all", "inherited_by", "is_private",
           "is_top_level", "module_depth"])
    check("and never rewrites the key it matched on",
          [c for c in ("release_id", "symbol_path", "kind", "sub_target")
           if f"{c} = EXCLUDED.{c}" in update], [])
    rels = [s for s in conn.sql if "INSERT INTO release" in s]
    check("releases with a known status carry it; the rest do not",
          ["analysis_status = EXCLUDED.analysis_status" in s for s in rels],
          [True, False])

    caps_old = {**caps, "analysis_status": False, "inherited_by": False}
    conn = FakeConn()
    with contextlib.redirect_stdout(io.StringIO()):
        db.write_all(conn, chg, usage, None, caps_old, None, rel)
    brk = next(s for s in conn.sql if "INSERT INTO breakage" in s)
    check("before migration 005 neither column is named at all",
          ("inherited_by" in brk,
           any("analysis_status" in s for s in conn.sql)), (False, False))


OLD_SRC = '''\
def f(a, b=1, sep=", "):
    pass


def old_api():
    """The old entry point.

    .. deprecated:: 1.0
       Use f instead.
    """
'''

NEW_SRC = '''\
def f(a, b=2, sep=" -> "):
    pass
'''


def case_detail() -> None:
    print("\n6. THE SENTENCE FIELDS THE API READS ARE WRITTEN")
    # Through the REAL ingest differ, so this is pinned to the text griffe
    # actually writes, not to a copy of it typed into a test.
    from ml.ingest.api_extract import diff_series

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-detail-"))
    try:
        for label, src in (("v0", OLD_SRC), ("v1", NEW_SRC)):
            (tmp / label / "pkg").mkdir(parents=True)
            (tmp / label / "pkg" / "__init__.py").write_text(src)
        rows = []
        for vf, vt, found in diff_series(
                "pkg", [("1.0", tmp / "v0"), ("1.1", tmp / "v1")], ["pkg"]):
            if isinstance(found, Exception):
                raise found
            rows += [{**r, "version_from": vf, "version_to": vt}
                     for r in found]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    detail = {(r.symbol, r.sub_target): db.detail_of(r)
              for r in pd.DataFrame(rows).fillna("").itertuples(index=False)}

    got = detail.get(("pkg.f", "b"), {})
    check("a changed default carries old_value and new_value",
          (got.get("old_value"), got.get("new_value")), ("1", "2"))
    got = detail.get(("pkg.f", "sep"), {})
    check("even when the default itself contains ' -> '",
          (got.get("old_value"), got.get("new_value")), ("', '", "' -> '"))
    got = detail.get(("pkg.old_api", ""), {})
    check("a removal the maintainer had marked deprecated says where",
          (got.get("deprecated_before"), got.get("was_deprecated_in")),
          (True, "1.0"))
    check("a removal carries no default values",
          "old_value" in got, False)

    text_false = pd.DataFrame([{**rows[0], "was_deprecated_before": "False"}])
    check('a CSV cell reading "False" is not a deprecation',
          "was_deprecated_in" in db.detail_of(
              next(text_false.itertuples(index=False))), False)
    check("an unsplittable message claims nothing (no guessed values)",
          db.default_change(
              "f(x): Parameter default was changed: a -> b -> c"), None)


def case_unlisted(chg: pd.DataFrame, rel: pd.DataFrame) -> None:
    print("\n7. A PACKAGE MISSING FROM packages.csv STILL GETS ITS ROWS")
    # The real shape on 25 Sep: packages.csv from one ingest run (here it
    # lists alpha and delta), changes.csv from two (beta and gamma have
    # changes but no packages.csv row).
    listed = pd.DataFrame({"package": ["alpha", "delta", "alpha"],
                           "download_rank": [1, 4, 1],
                           "github_repo": ["o/alpha", None, "o/alpha"]})
    rows = db.package_rows(chg, listed)
    check("every package with a change gets a row, listed or not",
          sorted(r["name"] for r in rows), ["alpha", "beta", "delta", "gamma"])
    check("a listed package keeps its packages.csv metadata",
          next(r for r in rows if r["name"] == "alpha")["github_repo"],
          "o/alpha")
    caps = {"sub_target": True, "inherited_by": True, "positive_rate": True,
            "analysis_status": True, "statuses": set(db.DB_STATUSES)}
    usage = pd.DataFrame({"symbol": ["alpha.f1"], "user_count": [4]})
    conn = FakeConn()
    with contextlib.redirect_stdout(io.StringIO()):
        db.write_all(conn, chg, usage, listed, caps, None, rel)
    check("and every one of its changes is sent to the database",
          len(conn.brk), len(chg))


def main() -> None:
    chg, rel = changes(), releases()
    check("load_releases collapses the duplicate delta 3.2.0 row",
          len(rel), 14)
    known, unknown, report = db.release_plan(chg, rel)
    known, unknown = by_key(known), by_key(unknown)
    case_agree(known, unknown, report)
    case_disagree(known, unknown, report)
    case_skipped(chg, rel)
    case_no_releases(chg)
    case_sql(chg, rel)
    case_detail()
    case_unlisted(chg, rel)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDo not run ml/db.py against Neon until this passes.")
        sys.exit(1)
    print("All checks passed. A re-load refreshes what it wrote, and every")
    print("release status comes from a file that agrees with changes.csv.")


if __name__ == "__main__":
    main()
