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
                 "in_dunder_all", "detail"]


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


def detail_of(row) -> dict:
    """The variable payload — the keys the API renders sentences from.

    Queryable attributes are typed columns; everything else lives here.
    `parameter` appears in both because sub_target is part of what makes a
    breakage unique AND the API wants it for the sentence.
    """
    d = {"griffe_message": _text(row, "explanation"),
         "griffe_kind": str(row.kind)}
    sub = _text(row, "sub_target")
    if sub:
        d["parameter" if str(row.kind).startswith("PARAMETER")
          else "removed_bases"] = sub
    return {k: v for k, v in d.items() if v}


def breakage_rows(changes: pd.DataFrame, release_id: dict):
    """Rows to insert, plus the natural key of each for the score join."""
    rows, keys = [], []
    for r in changes.itertuples(index=False):
        rid = release_id.get((r.package, r.version_to))
        if rid is None:
            continue
        sub = _text(r, "sub_target")
        rows.append({
            "release_id": rid, "symbol_path": r.symbol, "kind": r.kind,
            "sub_target": sub, "is_private": bool(r.is_private),
            "module_depth": int(r.module_depth),
            "is_top_level": bool(r.is_top_level),
            "in_dunder_all": bool(getattr(r, "in_dunder_all", False)),
            "detail": json.dumps(detail_of(r)),
        })
        keys.append((r.package, r.version_to, r.symbol, r.kind, sub))
    return rows, keys


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
    return create_engine(url, pool_pre_ping=True)


def check_schema(conn, allow_lossy: bool) -> bool:
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
    return has_sub


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


def write_all(conn, changes, usage, packages, has_sub, score_map) -> None:
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

    rows, keys = breakage_rows(changes, rel_id)
    cols = [c for c in BREAKAGE_COLS if has_sub or c != "sub_target"]
    if not has_sub:
        for r in rows:
            r.pop("sub_target", None)
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

    urows = (usage.rename(columns={"symbol": "symbol_path"})
             [["symbol_path", "user_count"]].to_dict("records"))
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
    conn.execute(text("""
        INSERT INTO model_run (version, pr_auc, precision_at_10, ndcg_at_20,
                               notes)
        VALUES (:version, :pr_auc, :precision_at_10, :ndcg_at_20, :notes)
        ON CONFLICT (version) DO UPDATE
            SET pr_auc = EXCLUDED.pr_auc,
                precision_at_10 = EXCLUDED.precision_at_10,
                ndcg_at_20 = EXCLUDED.ndcg_at_20,
                notes = EXCLUDED.notes, trained_at = now()
    """), run)
    print(f"  model_run     {run['version']}")

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


def dry_run(changes, usage) -> None:
    rels = release_rows(changes)
    fake = {(r["package"], r["version"]): 1 for r in rels}
    rows, keys = breakage_rows(changes, fake)

    print("would write (no database touched):")
    print(f"  package       {changes['package'].nunique():>7,}")
    print(f"  release       {len(rels):>7,}")
    print(f"  breakage      {len(rows):>7,}")
    print(f"  usage_index   {len(usage):>7,}")

    dupes = len(keys) - len(set(keys))
    print(f"\ndistinct (package, version, symbol, kind, sub_target): "
          f"{len(set(keys)):,}")
    if dupes:
        print(f"** {dupes:,} rows share a key and would collapse. Investigate "
              "before loading.")
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
        dry_run(changes, usage)
        return

    score_map = score_everything() if args.scores else None
    engine = connect()
    with engine.begin() as conn:
        has_sub = check_schema(conn, args.allow_missing_sub_target)
        write_all(conn, changes, usage, packages, has_sub, score_map)
    print("\ndone. Every write above is idempotent — rerun it any time.")


if __name__ == "__main__":
    main()
