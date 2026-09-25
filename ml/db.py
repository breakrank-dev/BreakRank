"""
Load the pipeline's CSVs into Postgres. The last piece before the site works.

    python ml/db.py --dry-run          # validate everything, touch nothing
    python ml/db.py                    # package, release, breakage, usage_index
    python ml/db.py --scores           # also model_run + prediction

Reads  data/packages.csv        -> package
       data/changes.csv         -> release, breakage
       data/releases.csv        -> release.analysis_status, n_changes, and
                                   the releases that changed nothing
                                   (optional; see release_plan)
       data/usage.csv           -> usage_index
       artifacts/metrics.json   -> model_run          (only with --scores)
       artifacts/ranker.txt     -> prediction         (only with --scores)

The database is the boundary between the two halves of this project. The
pipeline writes, the API reads, neither calls the other's code — which
makes the column meanings the interface. So this follows the frozen
contract (docs/api-contract.md, ml/contract.py) rather than doing whatever
looks reasonable.

FOUR RULES, all from the contract, all load-bearing.

1. EVERY WRITE IS IDEMPOTENT. A nightly job that dies at package 300 must
   be safe to rerun from the top, so every insert carries ON CONFLICT and
   a rerun is a no-op rather than an error or a duplicate.

2. model_run BEFORE prediction. prediction.model_version is a foreign key
   to model_run.version; the other order is rejected.

3. FACTS, NOT SENTENCES. The pipeline used to compose "pandas.DataFrame.
   append was removed. 412 packages call it." That is presentation, and
   storing it means rewording costs a re-ingest across thousands of
   releases instead of a two-minute API redeploy. griffe's message goes
   into `detail` as data; the API assembles the sentence.

4. PRIVATE SYMBOLS ARE STORED, NEVER DROPPED. The API hides them behind
   ?include_private=true. They are also the negative training examples —
   the "changed but nobody cares" rows the model exists to rank low.
   Dropping them here deletes the majority class.

This file never creates or alters a table. Migrations belong to the web
side (db/migrations). If the schema is not what the contract says, it
stops and explains.
"""

import argparse
import json
import os
import pathlib
import re
import sys

import pandas as pd
from packaging.version import InvalidVersion, Version

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.contract import bump_type  # noqa: E402
from ml.holdout import HOLDOUT_FILE, HOLDOUT_START  # noqa: E402

DATA = pathlib.Path("data")
ART = pathlib.Path("artifacts")
CHANGES, PACKAGES, USAGE = (DATA / "changes.csv", DATA / "packages.csv",
                            DATA / "usage.csv")
RELEASES = DATA / "releases.csv"

# The values release.analysis_status accepts, from Varad's migrations 005
# and 006. releases.csv also records `pre_release` and `dev_release`: the
# ingest filtered those out before analysing anything, and the database
# has no value for them, so they are skipped and counted rather than
# forced into a value that means something else.
DB_STATUSES = {"analysed", "analysed_clean", "analysis_failed", "no_source",
               "yanked", "no_baseline"}
# Releases that were part of a diffed chain. Only these can be a clean
# release's predecessor, which is what its bump_type is measured from.
CHAIN_STATUSES = {"analysed", "analysed_clean", "no_baseline",
                  "analysis_failed"}
FEATURES = DATA / "features.csv"
METRICS, RANKER = ART / "metrics.json", ART / "ranker.txt"

BATCH = 2000
BREAKAGE_COLS = ["release_id", "symbol_path", "kind", "sub_target",
                 "is_private", "module_depth", "is_top_level",
                 "in_dunder_all", "inherited_by", "detail"]

# Columns that exist only after a migration. Each is dropped from the
# insert when the database does not have it, so one loader serves both
# schemas and the two halves of this project deploy on their own
# schedules rather than in lockstep.
OPTIONAL_COLS = ("sub_target", "inherited_by")


# ------------------------------------------------------------ shaping rows

def load_frames():
    for p in (CHANGES, USAGE):
        if not p.exists():
            sys.exit(f"{p} not found — run the pipeline first.")
    packages = pd.read_csv(PACKAGES) if PACKAGES.exists() else None
    if packages is None:
        print(f"note: {PACKAGES} missing, so github_repo will be null. "
              "Rerun ml/ingest/run_ingest.py to produce it.\n")
    return pd.read_csv(CHANGES), pd.read_csv(USAGE), packages


def package_rows(changes: pd.DataFrame, packages) -> list[dict]:
    if packages is not None:
        p = packages.rename(columns={"package": "name"}).copy()
    else:
        p = (changes[["package", "package_rank"]].drop_duplicates("package")
             .rename(columns={"package": "name",
                              "package_rank": "download_rank"}).copy())
        p["github_repo"] = None
    # DEDUPLICATE ON NAME, for exactly the reason breakage_rows() does.
    # data/packages.csv is APPEND-ONLY: every resumed or retried run adds
    # its packages again, so after one 500-package run plus a 36-package
    # retry the file holds 532 rows for 500 names. Two rows with the same
    # name inside one multi-row INSERT ... ON CONFLICT DO UPDATE raise
    # "cannot affect row a second time" and take the whole transaction
    # with them — it is not a silent last-one-wins.
    #
    # keep="last" because a later run saw the package more recently: its
    # github_repo and download_rank come from the fresher PyPI response.
    p = p.drop_duplicates("name", keep="last")

    p = p[["name", "download_rank", "github_repo"]].astype(object)
    return p.where(pd.notna(p), None).to_dict("records")


def release_rows(changes: pd.DataFrame) -> list[dict]:
    """Every version seen, not only the ones that broke something.

    /analyze walks the range (current, latest] and needs a row for each
    release in it, including quiet ones. A version appearing only as a
    `version_from` still happened, so it goes in with a null date and null
    bump — an honest gap rather than an invented value.
    """
    later = (changes.groupby(["package", "version_to"])
             .agg(released_at=("released_at", "max"),
                  version_from=("version_from", "first"))
             .reset_index().rename(columns={"version_to": "version"}))
    later["bump_type"] = [bump_type(a, b) for a, b
                          in zip(later["version_from"], later["version"])]
    later = later[["package", "version", "released_at", "bump_type"]]

    seen = set(zip(later["package"], later["version"]))
    earlier = [{"package": p, "version": v, "released_at": None,
                "bump_type": None}
               for p, v in set(zip(changes["package"], changes["version_from"]))
               if (p, v) not in seen]

    out = pd.concat([later, pd.DataFrame(earlier)], ignore_index=True)
    return out.astype(object).where(pd.notna(out), None).to_dict("records")


def load_releases() -> pd.DataFrame | None:
    """data/releases.csv, or None for a dataset made before it existed."""
    if not RELEASES.exists():
        return None
    rel = pd.read_csv(RELEASES, dtype={"package": str, "version": str,
                                       "released_at": str, "status": str})
    # APPEND-ONLY, like packages.csv: a resumed or retried run writes its
    # packages again. The later row is the newer verdict, and two rows for
    # one release inside one upsert would abort the whole transaction.
    return rel.drop_duplicates(["package", "version"], keep="last")


def _version_order(v: str):
    try:
        return (0, Version(v))
    except InvalidVersion:
        return (1, v)


def release_plan(changes: pd.DataFrame, releases: pd.DataFrame | None,
                 allowed: set[str] = DB_STATUSES) -> tuple[list, list, dict]:
    """Every release row to write: (status known, status unknown, report).

    TWO SOURCES, trusted for different things.

      changes.csv   is the authority on every release that CHANGED
                    something: its status is `analysed` and n_changes is
                    its row count. That needs nothing but this file.
      releases.csv  is the only record of the releases that changed
                    nothing (analysed_clean) or were never compared
                    (no_baseline, analysis_failed, yanked, no_source).
                    NOTES §14 built it for this, and until 25 Sep nothing
                    loaded it: every release in the database read
                    `analysed` with n_changes 0, and a clean newest release
                    was not in the database at all.

    releases.csv is used for a package ONLY WHERE THE TWO FILES AGREE on
    it: the same analysed releases, with the same number of changes each.
    One ingest run writes both files, so a package they disagree on means
    releases.csv came from a DIFFERENT run, and loading its statuses would
    stamp one run's verdict onto another run's data. Those packages keep
    what the loader always did, and the report names them.

    A release with no known status goes in without one: a new row takes
    the database default and an existing row keeps what it has. Never
    `analysed_clean` by guesswork, which would have the site say "safe to
    upgrade" about a release nobody compared (NOTES §13.4).
    """
    base = {(r["package"], r["version"]): r for r in release_rows(changes)}
    counts = changes.groupby(["package", "version_to"]).size()
    known, unknown = {}, {}
    for key, r in base.items():
        if key in counts.index:
            known[key] = {**r, "analysis_status": "analysed",
                          "n_changes": int(counts[key])}
        else:
            unknown[key] = r

    in_changes = {p: dict(zip(g["version_to"], g["n"]))
                  for p, g in counts.rename("n").reset_index()
                  .groupby("package")}
    report = {"rows": 0, "agree": [], "disagree": {}, "skipped": {},
              "uncovered": sorted(in_changes)}
    if releases is None:
        return list(known.values()), list(unknown.values()), report

    analysed = releases[releases["status"] == "analysed"]
    n_said = pd.to_numeric(analysed["n_changes"], errors="coerce")
    in_releases = {p: dict(zip(g["version"], n_said.loc[g.index]))
                   for p, g in analysed.groupby("package")}
    agree, disagree = [], {}
    for p in sorted(set(releases["package"])):
        a, b = in_changes.get(p, {}), in_releases.get(p, {})
        only_a, only_b = set(a) - set(b), set(b) - set(a)
        recount = [v for v in set(a) & set(b) if b[v] != a[v]]
        if only_a or only_b or recount:
            disagree[p] = (len(only_a), len(only_b), len(recount))
        else:
            agree.append(p)

    use = releases[releases["package"].isin(agree)]
    storable = use["status"].isin(allowed)
    for p, g in use[storable].groupby("package"):
        chain = sorted(g.loc[g["status"].isin(CHAIN_STATUSES), "version"],
                       key=_version_order)
        before = dict(zip(chain[1:], chain[:-1]))
        for r in g.itertuples(index=False):
            key = (p, r.version)
            if key in known:
                continue
            prev = (before.get(r.version) if r.status == "analysed_clean"
                    else None)
            known[key] = {"package": p, "version": r.version,
                          "released_at": (r.released_at
                                          if isinstance(r.released_at, str)
                                          and r.released_at else None),
                          "bump_type": bump_type(prev, r.version) if prev
                          else None,
                          "analysis_status": r.status, "n_changes": 0}
            unknown.pop(key, None)

    skipped = use.loc[~storable, "status"].value_counts().to_dict()
    report.update(rows=len(releases), agree=agree, disagree=disagree,
                  skipped=skipped,
                  uncovered=sorted(set(in_changes) - set(releases["package"])))
    return list(known.values()), list(unknown.values()), report


def describe_plan(known: list, unknown: list, report: dict) -> str:
    """The release-status part of the load, in words, for both run modes."""
    if not report["rows"] and not report["agree"]:
        return ("  release status: no data/releases.csv, so only releases "
                "that changed something\n  get one (analysed, with their "
                "count). Nothing else is claimed.")
    statuses = pd.Series([r["analysis_status"] for r in known]).value_counts()
    lines = [
        f"  release status: releases.csv has {report['rows']:,} rows. It "
        f"agrees with changes.csv on {len(report['agree']):,} package(s),",
        f"  disagrees on {len(report['disagree']):,}, and "
        f"{len(report['uncovered']):,} package(s) with changes have no row "
        "in it.",
        "  statuses written: " + "   ".join(
            f"{s} {n:,}" for s, n in statuses.items()),
        f"  written with no status (existing rows keep theirs): "
        f"{len(unknown):,}",
    ]
    if report["skipped"]:
        lines.append("  skipped, the database has no value for them: " +
                     "   ".join(f"{s} {n:,}" for s, n in
                                report["skipped"].items()))
    if report["disagree"]:
        eg = list(report["disagree"].items())[:5]
        lines += [
            f"  ** releases.csv and changes.csv disagree on "
            f"{len(report['disagree']):,} package(s), so releases.csv is from",
            "  ** a different ingest run for those, and none of its statuses "
            "are used for them.",
            "  ** e.g. (analysed only in changes.csv, only in releases.csv, "
            "different counts):",
        ] + [f"  **   {p}: {a}, {b}, {c}" for p, (a, b, c) in eg]
    return "\n".join(lines)


def _text(row, field: str) -> str:
    """A CSV cell as a plain string, with pandas' NaN meaning empty.

    `x or ""` does NOT work here: an empty CSV cell reads back as float
    NaN, and NaN is TRUTHY, so `nan or ""` is nan and str() makes it the
    four-character word "nan". That shipped into a detail payload as
    {"removed_bases": "nan"} before the dry-run caught it, which would
    have put the word "nan" on the website.
    """
    v = getattr(row, field, "")
    return "" if v is None or v != v else str(v)


# griffe prefixes every explanation with where it found the change:
#   data/sdists/requests/2.34.0/.../adapters.py:193: HTTPAdapter.max_retries: ...
# That is a path on whichever laptop ran the ingest, inside a temp
# directory deleted seconds later. The API builds its user-facing sentence
# out of griffe_message, so left alone this puts a stranger's filesystem
# layout on the website and tells the reader nothing. Strip it at the
# boundary rather than at ingest: changes.csv stays a faithful record of
# what griffe said, and no re-ingest is needed to fix data already
# collected.
GRIFFE_LOCATION = re.compile(r"^\S*?\.pyi?:\d+:\s*")


def detail_of(row) -> dict:
    """The variable payload — the keys the API renders sentences from.

    Queryable attributes are typed columns; everything else lives here.
    `parameter` appears in both because sub_target is part of what makes a
    breakage unique AND the API wants it for the sentence.
    """
    d = {"griffe_message": GRIFFE_LOCATION.sub("", _text(row, "explanation")),
         "griffe_kind": str(row.kind)}
    sub = _text(row, "sub_target")
    if sub:
        d["parameter" if str(row.kind).startswith("PARAMETER")
          else "removed_bases"] = sub
    # ONLY WHEN TRUE, and that is the whole design. `detail` is already a
    # JSON column, so this adds no column and needs no migration — but a
    # key written on every row would be a claim on every row, and a
    # changes.csv from before 16 Sep cannot tell "we looked and found no
    # marker" apart from "we never looked". Writing the key only when the
    # answer is yes makes its ABSENCE mean nothing, which is honest for
    # both files. The API can render "the maintainer marked this
    # deprecated in {version_from}" when it is there and say nothing when
    # it is not.
    if bool(getattr(row, "was_deprecated_before", False)):
        d["deprecated_before"] = True
    return {k: v for k, v in d.items() if v}


def breakage_rows(changes: pd.DataFrame, release_id: dict):
    """Rows to insert, their natural keys, and what we dropped to get there.

    DEDUPLICATED ON THE DATABASE'S KEY, deliberately, because leaving it to
    Postgres does not do what it looks like it does. Two rows with the same
    key inside one multi-row INSERT ... ON CONFLICT DO UPDATE raise
    "ON CONFLICT DO UPDATE command cannot affect row a second time" — the
    whole transaction dies. It is not a silent last-one-wins.

    Measured on this dataset: exactly one collision in 23,025 rows.
    pygments.lexers.c_cpp.CppLexer.tokens is reported twice for the same
    release, once describing the 'keywords' part of that dict and once the
    'statements' part — CppLexer inherits `tokens` from CFamilyLexer, so
    griffe resolves it through two paths and reports both. They are
    genuinely different messages about the same attribute, and the schema
    has nowhere to put that distinction.

    So one of them is dropped, the count is printed, and nobody has to
    wonder later why the row count moved. Keeping the first is arbitrary
    and safe here: a lexer's internal token table is not something
    downstream code imports, so both rows are guaranteed negatives.
    """
    rows, keys, seen, dropped = [], [], set(), 0
    for r in changes.itertuples(index=False):
        rid = release_id.get((r.package, r.version_to))
        if rid is None:
            continue
        sub = _text(r, "sub_target")
        dbkey = (rid, r.symbol, r.kind, sub)
        if dbkey in seen:
            dropped += 1
            continue
        seen.add(dbkey)
        rows.append({
            "release_id": rid, "symbol_path": r.symbol, "kind": r.kind,
            "sub_target": sub, "is_private": bool(r.is_private),
            "module_depth": int(r.module_depth),
            "is_top_level": bool(r.is_top_level),
            "in_dunder_all": bool(getattr(r, "in_dunder_all", False)),
            # How many other classes inherit this exact change (NOTES
            # §10.2). Zero for ~97% of rows. A DISPLAY signal, not a model
            # feature — it has literally zero gain (§11.6) — but "this
            # change affects 3,242 inheriting classes" is exactly what a
            # human scanning a diff wants to know.
            "inherited_by": int(getattr(r, "inherited_by", 0) or 0),
            "detail": json.dumps(detail_of(r)),
        })
        keys.append((r.package, r.version_to, r.symbol, r.kind, sub))
    return rows, keys, dropped


# ----------------------------------------------------------------- schema

def connect():
    """Engine from DATABASE_URL. The URL is a secret and is never printed."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set.\n"
                 "Put it in .env, which is gitignored. Get it from Varad over "
                 "a private channel — never in a commit, an issue, or a "
                 "screenshot.")

    # Neon hands out "postgresql://...", and SQLAlchemy reads a bare
    # "postgresql://" as "use psycopg2" — the OLD driver, which we do not
    # install. The failure is ModuleNotFoundError: psycopg2, which reads
    # like a missing dependency rather than a URL-scheme mismatch and
    # sends you off installing the wrong package. Name the driver instead.
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg://" + url[len(prefix):]
            break

    return create_engine(url, pool_pre_ping=True)


def check_schema(conn, allow_lossy: bool) -> dict[str, bool]:
    from sqlalchemy import text

    have: dict[str, set[str]] = {}
    for t, c in conn.execute(text("""
            SELECT table_name, column_name FROM information_schema.columns
            WHERE table_schema = 'public'""")).fetchall():
        have.setdefault(t, set()).add(c)

    missing = {"package", "release", "breakage", "usage_index", "model_run",
               "prediction"} - have.keys()
    if missing:
        sys.exit(f"missing tables: {sorted(missing)}\n"
                 "Run db/apply_migrations.py, or ask Varad which migrations "
                 "are applied.")
    if "explanation" in have["breakage"]:
        sys.exit("breakage still has an `explanation` column — migration 003 "
                 "is not applied. This writer produces `detail` JSONB.")

    has_sub = "sub_target" in have["breakage"]
    if not has_sub:
        msg = ("breakage has no `sub_target` column — migration 004 is not "
               "applied.\nWithout it the key is (release_id, symbol_path, "
               "kind), and griffe emits\none row per changed PARAMETER, so "
               "rows sharing a symbol collapse.\nMeasured on this dataset: "
               "6.8% of rows silently lost, including 6 of\nthe 7 "
               "typing_extensions.TypedDict parameter removals.\n")
        if not allow_lossy:
            sys.exit(msg + "\nAsk Varad to land migration 004, or rerun with "
                     "--allow-missing-sub-target\nto load a knowingly lossy "
                     "copy for testing.")
        print("!" * 68 + f"\n{msg}Loading anyway because you asked.\n"
              + "!" * 68 + "\n")

    # Migration 005 (15 Sep) added inherited_by to breakage and
    # positive_rate to model_run. DETECTED, NOT ASSUMED: this loader runs
    # against whatever is deployed, and naming a column that does not
    # exist fails the whole transaction. Writing them when they are there
    # and staying quiet when they are not is what lets the two halves of
    # this project deploy on their own schedules.
    caps = {
        "sub_target": has_sub,
        "inherited_by": "inherited_by" in have["breakage"],
        "positive_rate": "positive_rate" in have["model_run"],
        "analysis_status": {"analysis_status", "n_changes"} <= have["release"],
    }
    for name in ("inherited_by", "positive_rate", "analysis_status"):
        if not caps[name]:
            print(f"  note: {name} not present — migration 005 is not applied,"
                  f"\n        so that value is not written. Everything else "
                  "loads normally.")

    # WHICH STATUSES THIS DATABASE ACCEPTS, read from its own CHECK
    # constraint rather than assumed. Migration 006 added no_baseline; a
    # database without it would reject the whole load on the first such
    # row, so the loader writes only what the constraint allows.
    caps["statuses"] = set(DB_STATUSES)
    found = conn.execute(text("""
        SELECT pg_get_constraintdef(oid) FROM pg_constraint
        WHERE conrelid = 'release'::regclass
          AND pg_get_constraintdef(oid) LIKE '%analysis_status%'""")).scalar()
    if found:
        caps["statuses"] = set(re.findall(r"'([a-z_]+)'", found))
    return caps


# ------------------------------------------------------------------ writes

def upsert_ids(conn, sql: str, rows: list[dict]) -> list:
    """One row at a time, RETURNING id. Fine for hundreds, not thousands."""
    from sqlalchemy import text

    stmt = text(sql)
    return [(conn.execute(stmt, r).fetchone() or [None])[0] for r in rows]


def executemany(conn, sql: str, rows: list[dict]) -> None:
    from sqlalchemy import text

    stmt = text(sql)
    for i in range(0, len(rows), BATCH):
        conn.execute(stmt, rows[i:i + BATCH])


def write_all(conn, changes, usage, packages, caps, score_map,
              releases=None) -> None:
    from sqlalchemy import text

    pkgs = package_rows(changes, packages)
    ids = upsert_ids(conn, """
        INSERT INTO package (name, download_rank, github_repo)
        VALUES (:name, :download_rank, :github_repo)
        ON CONFLICT (name) DO UPDATE
            SET download_rank = EXCLUDED.download_rank,
                github_repo = COALESCE(EXCLUDED.github_repo,
                                       package.github_repo)
        RETURNING id""", pkgs)
    pkg_id = {r["name"]: i for r, i in zip(pkgs, ids) if i}
    print(f"  package       {len(pkg_id):>7,}")

    known, unknown, plan = release_plan(changes, releases, caps["statuses"])
    known = [r for r in known if r["package"] in pkg_id]
    unknown = [r for r in unknown if r["package"] in pkg_id]
    for r in known + unknown:
        r["package_id"] = pkg_id[r["package"]]
    upsert = """
        INSERT INTO release (package_id, version, released_at, bump_type{c})
        VALUES (:package_id, :version, :released_at, :bump_type{v})
        ON CONFLICT (package_id, version) DO UPDATE
            SET released_at = COALESCE(EXCLUDED.released_at,
                                       release.released_at),
                bump_type = COALESCE(EXCLUDED.bump_type, release.bump_type){s}
    """
    if caps["analysis_status"]:
        executemany(conn, upsert.format(
            c=", analysis_status, n_changes",
            v=", :analysis_status, :n_changes",
            s=",\n                analysis_status = EXCLUDED.analysis_status,"
              "\n                n_changes = EXCLUDED.n_changes"), known)
        rest = unknown
    else:
        rest = known + unknown
    executemany(conn, upsert.format(c="", v="", s=""), rest)
    # Read the ids back in one query instead of one RETURNING per row: with
    # the clean releases added this is thousands of rows, and each round
    # trip to Neon costs tens of milliseconds.
    rel_id = {(name, ver): rid for rid, name, ver in conn.execute(text("""
        SELECT r.id, p.name, r.version FROM release r
        JOIN package p ON p.id = r.package_id""")).fetchall()}
    print(f"  release       {len(known) + len(unknown):>7,}")
    print(describe_plan(known, unknown, plan) if caps["analysis_status"] else
          "  release status: none written, this database has no "
          "analysis_status column\n  (migration 005).")

    has_sub = caps["sub_target"]
    rows, keys, dropped = breakage_rows(changes, rel_id)
    # Drop any optional column this database does not have, from both the
    # column list and the row dicts — a bound parameter with no column to
    # land in is as fatal as a column with no parameter.
    cols = [c for c in BREAKAGE_COLS
            if c not in OPTIONAL_COLS or caps.get(c)]
    for c in OPTIONAL_COLS:
        if not caps.get(c):
            for r in rows:
                r.pop(c, None)
    conflict = "release_id, symbol_path, kind" + (", sub_target" if has_sub
                                                  else "")
    # A RE-LOAD REFRESHES EVERY COLUMN IT WROTE, not only `detail`. Until
    # 25 Sep only detail was updated, so any value computed after a row's
    # first load never reached the database: inherited_by (the fold came
    # on 12 Sep, after the 6 Sep load, and migration 005 then set every
    # existing row to 0), and is_private, module_depth, is_top_level and
    # in_dunder_all whenever their rules change. Measured on Neon on 25 Sep:
    # 18 rows with inherited_by > 0 against 647 in changes.csv.
    key_cols = {"release_id", "symbol_path", "kind", "sub_target"}
    refresh = ",\n                ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c not in key_cols)
    # executemany, not RETURNING per row: 23,000 round trips is minutes of
    # latency for data we can read back in one SELECT.
    executemany(conn, f"""
        INSERT INTO breakage ({', '.join(cols)})
        VALUES ({', '.join(':' + c for c in cols)})
        ON CONFLICT ({conflict}) DO UPDATE
            SET {refresh}
    """, rows)

    sub_sel = "sub_target" if has_sub else "''"
    found = conn.execute(text(f"""
        SELECT b.id, r.package_id, p.name, r.version, b.symbol_path, b.kind,
               {sub_sel}
        FROM breakage b
        JOIN release r ON r.id = b.release_id
        JOIN package p ON p.id = r.package_id
    """)).fetchall()
    br_id = {(name, ver, sym, kind, sub or ""): bid
             for bid, _, name, ver, sym, kind, sub in found}
    print(f"  breakage      {len(rows):>7,} sent, {len(br_id):,} in table")
    if len(br_id) < len(set(keys)):
        print(f"  ** {len(set(keys)) - len(br_id):,} distinct changes did not "
              "survive the uniqueness key.\n  ** That is the sub_target "
              "problem — see migration 004.")

    # Same guard, same reason. usage.csv has never carried a duplicate
    # symbol, but it is written by a resumable scanner and nothing in the
    # schema stops it, so this costs nothing and removes a way for one
    # transaction to die at row 38,000.
    urows = (usage.rename(columns={"symbol": "symbol_path"})
             [["symbol_path", "user_count"]]
             .drop_duplicates("symbol_path", keep="last")
             .to_dict("records"))
    executemany(conn, """
        INSERT INTO usage_index (symbol_path, user_count, computed_at)
        VALUES (:symbol_path, :user_count, now())
        ON CONFLICT (symbol_path) DO UPDATE
            SET user_count = EXCLUDED.user_count, computed_at = now()
    """, urows)
    print(f"  usage_index   {len(urows):>7,}")

    if score_map is None:
        print("\nNo predictions written (rerun with --scores). The API falls "
              "back to\nordering by user_count, so the site still works — it "
              "ranks by\npopularity instead of by the model.")
        return

    # RULE 2: model_run must exist before anything references its version.
    run = json.loads(METRICS.read_text())

    # THE FLOOR, AS A COLUMN. PR-AUC's floor is the positive rate, so a
    # stored pr_auc without it cannot be read by anyone who was not in the
    # room (§1, day 5). It has been inside the `notes` string since then,
    # recoverable only by parsing prose. Migration 005 gave it a column.
    # The API contract (decision 13) now forbids displaying one without
    # the other, so this is the value that rule depends on.
    # A metrics.json written before 16 Sep has the floor only inside the
    # notes prose. Parse it out rather than writing NULL — a file produced
    # by an older train.py is still a valid model run, and silently
    # storing an unreadable score is the exact failure this column exists
    # to end.
    if run.get("positive_rate") is None:
        found = re.search(r"positive_rate=([0-9.]+)", run.get("notes", "") or "")
        run["positive_rate"] = float(found.group(1)) if found else None
        if found:
            print(f"  note: positive_rate recovered from the notes string "
                  f"({run['positive_rate']}).\n        Re-run "
                  "ml/model/train.py to write it as a field.")
        elif caps.get("positive_rate"):
            print("  ** metrics.json has no positive_rate and none could be "
                  "parsed from\n  ** notes. pr_auc will be stored without its "
                  "floor, which makes it\n  ** unreadable. Re-run "
                  "ml/model/train.py before serving this run.")
    extra = ", positive_rate" if caps.get("positive_rate") else ""
    extra_val = ", :positive_rate" if caps.get("positive_rate") else ""
    extra_set = ("\n                positive_rate = EXCLUDED.positive_rate,"
                 if caps.get("positive_rate") else "")

    # trained_at IS NOT TOUCHED ON CONFLICT, and that is a fix rather than
    # an omission. It used to be set to now() on every update, so
    # re-running --scores against an OLDER model version would stamp it as
    # the newest — and the API picks the current model with
    # `ORDER BY trained_at DESC LIMIT 1`. One re-score of a superseded run
    # would have put it back on the site.
    #
    # This is the mirror of the bug Varad found on his side: his seed used
    # DEFAULT now(), so the fixture silently became newest on every reseed.
    # Same bug, opposite direction, both fixed before there were two real
    # models to confuse.
    conn.execute(text(f"""
        INSERT INTO model_run (version, pr_auc, precision_at_10, ndcg_at_20,
                               notes{extra})
        VALUES (:version, :pr_auc, :precision_at_10, :ndcg_at_20,
                :notes{extra_val})
        ON CONFLICT (version) DO UPDATE
            SET pr_auc = EXCLUDED.pr_auc,
                precision_at_10 = EXCLUDED.precision_at_10,
                ndcg_at_20 = EXCLUDED.ndcg_at_20,{extra_set}
                notes = EXCLUDED.notes
    """), run)
    pr = run.get("positive_rate")
    print(f"  model_run     {run['version']}"
          + (f"   positive_rate {pr}" if caps.get("positive_rate") and pr
             else ""))

    preds = [{"breakage_id": br_id[k], "model_version": run["version"],
              "score": float(s)}
             for k, s in score_map.items() if k in br_id]
    executemany(conn, """
        INSERT INTO prediction (breakage_id, model_version, score, computed_at)
        VALUES (:breakage_id, :model_version, :score, now())
        ON CONFLICT (breakage_id, model_version) DO UPDATE
            SET score = EXCLUDED.score, computed_at = now()
    """, preds)
    unmatched = len(score_map) - len(preds)
    print(f"  prediction    {len(preds):>7,}"
          + (f"   ({unmatched:,} scores had no breakage row)"
             if unmatched else ""))


# ----------------------------------------------------------------- scoring

def score_everything() -> dict:
    """Score EVERY row, not just the test half.

    Training measures on held-out data; serving needs a score for every
    breakage in the database, including rows the model trained on. Those
    are different things, and conflating them is how training scores end
    up being reported as results. The metrics in model_run come from the
    test half; these scores cover everything.
    """
    import lightgbm as lgb

    from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC

    for p in (FEATURES, RANKER, METRICS):
        if not p.exists():
            sys.exit(f"{p} not found — run ml/model/train.py first.")

    # Version columns read as text. They are half of every prediction key,
    # and a file whose versions all look like numbers ("2.10", "3.1") would
    # otherwise come back as floats and match nothing, silently.
    as_text = {"version_from": str, "version_to": str}
    df = pd.read_csv(FEATURES, dtype=as_text)
    # SERVING IS NOT EVALUATING. Since the holdout froze (ml/holdout.py),
    # features.csv holds only rows released before 2026-08-04, and the
    # newest releases, the ones a visitor most wants ranked, live in
    # holdout.csv. Scoring them reads no label, so the site keeps them.
    # What must not happen is the reverse: joining these scores to the
    # usage table to compute a metric before the final report.
    if HOLDOUT_FILE.exists():
        held = pd.read_csv(HOLDOUT_FILE, dtype=as_text)
        # An EMPTY holdout.csv (an older dataset) is header-only, and
        # concatenating it turns every feature column to object dtype,
        # which LightGBM refuses. Nothing to add, so add nothing.
        if len(held):
            df = pd.concat([df, held], ignore_index=True)
    else:
        print(f"note: {HOLDOUT_FILE} not found, so releases from "
              f"{HOLDOUT_START.date()} on get no score this run. "
              "Rebuild with ml/features/build.py.")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    raw = lgb.Booster(model_file=str(RANKER)).predict(
        df[NUMERIC + BOOLEAN + CATEGORICAL])
    df["score"] = raw[:, 1] if getattr(raw, "ndim", 1) > 1 else raw

    return {(r.package, r.version_to, r.symbol, r.kind,
             _text(r, "sub_target")): r.score
            for r in df.itertuples(index=False)}


def dry_run(changes, usage, packages, releases=None) -> None:
    # Filtered to the packages the writer will create, as write_all does,
    # so the preview counts the same rows the load will write.
    names = {p["name"] for p in package_rows(changes, packages)}
    known, unknown, plan = release_plan(changes, releases)
    known = [r for r in known if r["package"] in names]
    unknown = [r for r in unknown if r["package"] in names]
    rels = known + unknown
    # A DISTINCT id per release. It was `: 1` for every one of them, which
    # was harmless until breakage_rows started deduplicating on release_id
    # — then all 2,061 releases shared an id, every symbol that changed in
    # two releases looked like a collision, and the preview reported 2,702
    # drops against a true figure of 1. A stub that stops being a stub is
    # worse than no stub.
    fake = {(r["package"], r["version"]): i for i, r in enumerate(rels, 1)}
    rows, keys, dropped = breakage_rows(changes, fake)

    # Must go through package_rows(), not changes['package'].nunique().
    # Those two numbers are different and the difference is the point: the
    # writer loads every package in packages.csv, INCLUDING the ones that
    # produced no breaking change, because the package table's job is to
    # record that we looked. Counting distinct packages in changes.csv
    # gave 410 here against a real write of 500, so the preview quietly
    # disagreed with the thing it exists to preview.
    pkgs = package_rows(changes, packages)
    analysed, broke = len(pkgs), changes["package"].nunique()
    raw = 0 if packages is None else len(packages)
    if raw > analysed:
        print(f"note: data/packages.csv holds {raw:,} rows for {analysed:,} "
              f"packages — {raw - analysed} duplicate(s) from resumed or\n"
              "retried runs, collapsed before insert. Left in place they "
              "would abort the\nwhole transaction, not overwrite quietly.\n")

    print("would write (no database touched):")
    print(f"  package       {analysed:>7,}")
    print(f"  release       {len(rels):>7,}")
    print(describe_plan(known, unknown, plan))
    print(f"  breakage      {len(rows):>7,}   (a row already in the database "
          "gets every column\n                         refreshed, "
          "inherited_by included)")
    print(f"  usage_index   {len(usage):>7,}")

    if analysed > broke:
        print(f"\n{analysed - broke} of those {analysed} packages produced no "
              f"breaking change at all.\nThey are written anyway: 'analysed, "
              "found nothing' has to be\ndistinguishable from 'never "
              "analysed' (decision 1).")

    print(f"\ndistinct (release, symbol, kind, sub_target): {len(rows):,}")
    if dropped:
        print(f"{dropped:,} row(s) dropped as duplicate keys before insert — "
              "griffe\nreported the same symbol twice for one release with no "
              "sub_target to\ntell them apart. Left to Postgres this is an "
              "error, not an overwrite,\nso it is resolved here and counted. "
              "See breakage_rows().")
    else:
        print("no key collisions — every change survives the insert.")

    priv = int(changes["is_private"].sum())
    print(f"private rows kept: {priv:,} ({priv / len(changes):.1%}) — stored "
          "and hidden, never dropped")
    print(f"\nexample detail payload:\n  "
          f"{json.dumps(detail_of(next(changes.itertuples(index=False))))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the pipeline into Postgres.")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report, connect to nothing")
    ap.add_argument("--scores", action="store_true",
                    help="also write model_run and prediction")
    ap.add_argument("--allow-missing-sub-target", action="store_true",
                    help="load a knowingly lossy copy before migration 004")
    args = ap.parse_args()

    changes, usage, packages = load_frames()
    releases = load_releases()
    print(f"{len(changes):,} changes   {len(usage):,} used symbols   "
          f"{changes['package'].nunique()} packages\n")

    if args.dry_run:
        dry_run(changes, usage, packages, releases)
        return

    score_map = score_everything() if args.scores else None
    engine = connect()
    with engine.begin() as conn:
        caps = check_schema(conn, args.allow_missing_sub_target)
        write_all(conn, changes, usage, packages, caps, score_map, releases)
    print("\ndone. Every write above is idempotent — rerun it any time.")


if __name__ == "__main__":
    main()
