"""
Load the pipeline's CSVs into Postgres. The last piece before the site works.

    python ml/db.py --dry-run          # validate everything, touch nothing
    python ml/db.py                    # package, release, breakage, usage_index
    python ml/db.py --scores           # also model_run + prediction

Reads  data/packages.csv        -> package
       data/changes.csv         -> release, breakage
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.contract import bump_type  # noqa: E402

DATA = pathlib.Path("data")
ART = pathlib.Path("artifacts")
CHANGES, PACKAGES, USAGE = (DATA / "changes.csv", DATA / "packages.csv",
                            DATA / "usage.csv")
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
    }
    for name in ("inherited_by", "positive_rate"):
        if not caps[name]:
            print(f"  note: {name} not present — migration 005 is not applied,"
                  f"\n        so that value is not written. Everything else "
                  "loads normally.")
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


def write_all(conn, changes, usage, packages, caps, score_map) -> None:
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

    rels = [r for r in release_rows(changes) if r["package"] in pkg_id]
    for r in rels:
        r["package_id"] = pkg_id[r["package"]]
    ids = upsert_ids(conn, """
        INSERT INTO release (package_id, version, released_at, bump_type)
        VALUES (:package_id, :version, :released_at, :bump_type)
        ON CONFLICT (package_id, version) DO UPDATE
            SET released_at = COALESCE(EXCLUDED.released_at,
                                       release.released_at),
                bump_type = COALESCE(EXCLUDED.bump_type, release.bump_type)
        RETURNING id""", rels)
    rel_id = {(r["package"], r["version"]): i
              for r, i in zip(rels, ids) if i}
    print(f"  release       {len(rel_id):>7,}")

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
    # executemany, not RETURNING per row: 23,000 round trips is minutes of
    # latency for data we can read back in one SELECT.
    executemany(conn, f"""
        INSERT INTO breakage ({', '.join(cols)})
        VALUES ({', '.join(':' + c for c in cols)})
        ON CONFLICT ({conflict}) DO UPDATE SET detail = EXCLUDED.detail
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

    df = pd.read_csv(FEATURES)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    raw = lgb.Booster(model_file=str(RANKER)).predict(
        df[NUMERIC + BOOLEAN + CATEGORICAL])
    df["score"] = raw[:, 1] if getattr(raw, "ndim", 1) > 1 else raw

    return {(r.package, r.version_to, r.symbol, r.kind,
             _text(r, "sub_target")): r.score
            for r in df.itertuples(index=False)}


def dry_run(changes, usage, packages) -> None:
    rels = release_rows(changes)
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
    print(f"  breakage      {len(rows):>7,}")
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
    print(f"{len(changes):,} changes   {len(usage):,} used symbols   "
          f"{changes['package'].nunique()} packages\n")

    if args.dry_run:
        dry_run(changes, usage, packages)
        return

    score_map = score_everything() if args.scores else None
    engine = connect()
    with engine.begin() as conn:
        caps = check_schema(conn, args.allow_missing_sub_target)
        write_all(conn, changes, usage, packages, caps, score_map)
    print("\ndone. Every write above is idempotent — rerun it any time.")


if __name__ == "__main__":
    main()
