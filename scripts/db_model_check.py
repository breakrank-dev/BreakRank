"""
Which model will the site serve?

    python scripts/db_model_check.py
    python scripts/db_model_check.py --ranker model

READ ONLY. Every statement in this file is a SELECT, so it is safe to run
against the live database at any time.

WHY THIS EXISTS. The API does not ask for the newest model on every
request. It picks one ONCE, when it starts: the model_run row with the
newest trained_at, unless the MODEL_VERSION setting pins one (decision 8,
api/config.py on main). It reads that model's scores on every request,
so a load under the name it already serves is live at once, with no
restart. Two things can still leave the site on an old model after
`ml/db.py --scores` has loaded a new one:

  1. The load added a new version, and the API has not restarted since.
     It keeps serving what it picked when it last started.
  2. The version was already in the table from an earlier load. The
     loader never moves trained_at on a re-load (NOTES §13.5, so that
     re-scoring an old model cannot make it current by accident), and so
     a re-loaded version keeps its old date and may not be the newest.

A MODEL_VERSION pin overrides both. The API's /health answer names the
model it is serving now; this prints the command that asks it.

This prints every model_run row, newest first, with how many predictions
each holds, and says which one a restarted API would serve and whether
that is the ranker that ships. From 11 Oct that is path length alone
(NOTES §36.2), whose row is artifacts/path_metrics.json; --ranker model
asks the same of the model, artifacts/metrics.json, the fallback.

Then, for that ranker, the day each of its scores was written, and how
many version-string rows have one. Added 1 Oct,
after this check found 5,686 of lambdarank-label_alias's scores left over
from an earlier load beside 21,498 new ones: before that day's ml/db.py
fix, a re-load only overwrote the rows it scored. A version's scores
should all come from one load, and F1 took the version-string rows out of
the model, so after a load the first list has one day and the count is 0.
F39 (NOTES §32) took package-metadata rows out the same way, so they are
counted the same way, from 9 Oct: 0 after the first load that follows.

Both counts use build.py's own rules, version_strings() and
metadata_strings(). Since 10 Oct F1 names a version constant's changed
value only, and the bare names (version, VERSION, version_tuple) only
directly in a module, which it reads from griffe's explanation; and
neither F1 nor F39 takes a value changed to `unset` (NOTES §33.1). The
database keeps that explanation as detail.griffe_message, so the same
functions decide here.

From 12 Oct the same two rules are counted over the WHOLE breakage table
as well, scored or not (NOTES §33.5): the loader no longer writes such
rows, and scripts/db_drop_versions.py removes the ones earlier loads
wrote, so after it both numbers read 0. Varad's breakrank-fixture rows are
left out of that count, as that script leaves them alone.
"""

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import FIXTURE_PACKAGE, connect  # noqa: E402
from ml.features.build import (METADATA_LEAVES, VERSION_KIND,  # noqa: E402
                               VERSION_LEAVES, metadata_strings,
                               version_strings)
from ml.model.ship import PATH_METRICS, RANKERS, SHIPPED_RANKER  # noqa: E402

METRICS = pathlib.Path("artifacts") / "metrics.json"
# /health answers {"ok": true, "model_version": ...}, the model the API is
# serving now. The address is docs/api-contract.md's, on main.
HEALTH = "https://breakrank.onrender.com/health"


def main() -> None:
    from sqlalchemy import text

    ap = argparse.ArgumentParser(description="Which model will the site "
                                 "serve? Read only.")
    ap.add_argument("--ranker", choices=RANKERS, default=SHIPPED_RANKER,
                    help="whose row to check for: path, path length "
                         "alone, the ranker that ships (NOTES §36.2); or "
                         "model, the fallback")
    args = ap.parse_args()
    metrics = PATH_METRICS if args.ranker == "path" else METRICS
    load = ("python ml/db.py --scores" if args.ranker == SHIPPED_RANKER
            else f"python ml/db.py --scores --ranker {args.ranker}")
    ours = (json.loads(metrics.read_text()).get("version")
            if metrics.exists() else None)
    with connect().connect() as conn:
        runs = conn.execute(text("""
            SELECT m.version, m.trained_at, m.pr_auc,
                   (SELECT count(*) FROM prediction p
                    WHERE p.model_version = m.version) AS predictions
            FROM model_run m
            ORDER BY m.trained_at DESC NULLS LAST, m.version
        """)).fetchall()
        breakages = conn.execute(
            text("SELECT count(*) FROM breakage")).scalar()
        days = conn.execute(text("""
            SELECT date(computed_at) AS day, count(*) FROM prediction
            WHERE model_version = :v GROUP BY 1 ORDER BY 1
        """), {"v": ours}).fetchall() if ours else []
        # The candidates by name and kind, scored by this model. build.py's
        # own rules then read griffe's explanation: where a bare version
        # name sits, and whether a value was changed to unset (§33.1).
        named = conn.execute(text(r"""
            SELECT b.symbol_path, b.kind, b.detail ->> 'griffe_message'
            FROM prediction p
            JOIN breakage b ON b.id = p.breakage_id
            WHERE p.model_version = :v
              AND b.kind = :kind
              AND regexp_replace(b.symbol_path, '^.*\.', '') = ANY(:leaves)
        """), {"v": ours, "kind": VERSION_KIND,
               "leaves": sorted(VERSION_LEAVES | METADATA_LEAVES)}
        ).fetchall() if ours else []
        # The same candidates over the whole table, scored or not (§33.5),
        # Varad's fixture rows left out.
        in_table = conn.execute(text(r"""
            SELECT b.symbol_path, b.kind, b.detail ->> 'griffe_message'
            FROM breakage b
            JOIN release r ON r.id = b.release_id
            JOIN package p ON p.id = r.package_id
            WHERE b.kind = :kind
              AND regexp_replace(b.symbol_path, '^.*\.', '') = ANY(:leaves)
              AND p.name <> :fixture
        """), {"kind": VERSION_KIND,
               "leaves": sorted(VERSION_LEAVES | METADATA_LEAVES),
               "fixture": FIXTURE_PACKAGE}).fetchall()

    cols = ["symbol", "kind", "explanation"]
    scored = pd.DataFrame([tuple(r) for r in named], columns=cols)
    stale_vs = int(version_strings(scored).sum())
    stale_meta = int(metadata_strings(scored).sum())
    whole = pd.DataFrame([tuple(r) for r in in_table], columns=cols)
    table_vs = int(version_strings(whole).sum())
    table_meta = int((metadata_strings(whole) & ~version_strings(whole)).sum())

    print(f"\nmodel_run rows, newest first ({breakages:,} breakage rows in "
          "the table):\n")
    print(f"  {'version':<34}{'trained_at':<28}{'pr_auc':>8}"
          f"{'predictions':>13}")
    for version, at, pr, n in runs:
        mark = f"   <- {metrics}" if version == ours else ""
        print(f"  {version:<34}{str(at)[:26]:<28}"
              f"{'-' if pr is None else f'{pr:.4f}':>8}{n:>13,}{mark}")

    if days:
        print(f"\n{ours}'s scores, by the day they were written:")
        for day, n in days:
            print(f"  {day}   {n:>8,}")
        if len(days) > 1:
            print("** These come from more than one load, and can be more "
                  "than one model's,\n** ranked together on the site. Re-run "
                  f"{load} with the 1 Oct loader:\n** it "
                  "replaces the version's scores whole.")
        print(f"version-string rows with a score: {stale_vs:,}"
              + ("   (none, as F1 intends)" if not stale_vs else
                 "   ** F1 took these out of the model; these scores are "
                 "an earlier model's"))
        print(f"package-metadata rows with a score: {stale_meta:,}"
              + ("   (none, as F39 intends)" if not stale_meta else
                 "   ** F39 took these out of the model; reload:\n** "
                 f"{load}"))

    print(f"\nin the whole breakage table, scored or not ({FIXTURE_PACKAGE}'s "
          "rows left out):")
    print(f"  version-string rows {table_vs:,}   package-metadata rows "
          f"{table_meta:,}"
          + ("   (none; the loader writes none, NOTES §33.5)"
             if not table_vs + table_meta else
             "\n  ** the loader no longer writes these; "
             "scripts/db_drop_versions.py removes them,\n  ** dry run "
             "first (NOTES §33.5)"))

    if not runs:
        print(f"  (none) Load one with: {load}")
        return
    newest = runs[0][0]
    ask = f"To see the model it serves now:  curl -s {HEALTH}"
    print()
    if ours is None:
        print(f"A restarted API would serve {newest}. There is no "
              f"{metrics} here to compare it with.")
        print(ask)
    elif newest == ours:
        print(f"A restarted API serves {ours}, the one in {metrics}.")
        print(ask)
        print("It reads that model's scores on every request, so if that "
              f"names\n{ours}, the scores loaded under it are live, with no "
              "restart.\nIf it names another model, restart the API. If it "
              "still does after that,\nMODEL_VERSION is pinning it: set "
              f"MODEL_VERSION={ours} where\nthe API runs, or remove it, "
              "and restart again.")
    elif args.ranker != SHIPPED_RANKER:
        print(f"A restarted API would serve {newest}, not {ours}, the "
              f"fallback in {metrics},\nwhich is not meant to be served "
              "(NOTES §36.2).")
        print(ask)
    else:
        print(f"** A restarted API would serve {newest}, NOT {ours}, the "
              f"one in\n** {metrics}: {newest} has the newer trained_at.")
        print(f"** Fix without touching data: set MODEL_VERSION={ours} "
              "where the API runs,\n** and restart it. That setting exists "
              "for exactly this (api/config.py on main).")
        print(f"** {ask}")


if __name__ == "__main__":
    main()
