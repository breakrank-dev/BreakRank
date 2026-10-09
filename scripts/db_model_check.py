"""
Which model will the site serve?

    python scripts/db_model_check.py

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
that is the model in artifacts/metrics.json.

Then, for the model in artifacts/metrics.json, the day each of its scores
was written, and how many version-string rows have one. Added 1 Oct,
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
"""

import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import connect  # noqa: E402
from ml.features.build import (METADATA_LEAVES, VERSION_KIND,  # noqa: E402
                               VERSION_LEAVES, metadata_strings,
                               version_strings)

METRICS = pathlib.Path("artifacts") / "metrics.json"
# /health answers {"ok": true, "model_version": ...}, the model the API is
# serving now. The address is docs/api-contract.md's, on main.
HEALTH = "https://breakrank.onrender.com/health"


def main() -> None:
    from sqlalchemy import text

    ours = (json.loads(METRICS.read_text()).get("version")
            if METRICS.exists() else None)
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

    scored = pd.DataFrame([tuple(r) for r in named],
                          columns=["symbol", "kind", "explanation"])
    stale_vs = int(version_strings(scored).sum())
    stale_meta = int(metadata_strings(scored).sum())

    print(f"\nmodel_run rows, newest first ({breakages:,} breakage rows in "
          "the table):\n")
    print(f"  {'version':<34}{'trained_at':<28}{'pr_auc':>8}"
          f"{'predictions':>13}")
    for version, at, pr, n in runs:
        mark = "   <- artifacts/metrics.json" if version == ours else ""
        print(f"  {version:<34}{str(at)[:26]:<28}"
              f"{'-' if pr is None else f'{pr:.4f}':>8}{n:>13,}{mark}")

    if days:
        print(f"\n{ours}'s scores, by the day they were written:")
        for day, n in days:
            print(f"  {day}   {n:>8,}")
        if len(days) > 1:
            print("** These come from more than one load, and can be more "
                  "than one model's,\n** ranked together on the site. Re-run "
                  "python ml/db.py --scores with the 1 Oct loader:\n** it "
                  "replaces the version's scores whole.")
        print(f"version-string rows with a score: {stale_vs:,}"
              + ("   (none, as F1 intends)" if not stale_vs else
                 "   ** F1 took these out of the model; these scores are "
                 "an earlier model's"))
        print(f"package-metadata rows with a score: {stale_meta:,}"
              + ("   (none, as F39 intends)" if not stale_meta else
                 "   ** F39 took these out of the model; reload:\n** "
                 "python ml/db.py --scores"))

    if not runs:
        print("  (none) Load one with: python ml/db.py --scores")
        return
    newest = runs[0][0]
    ask = f"To see the model it serves now:  curl -s {HEALTH}"
    print()
    if ours is None:
        print(f"A restarted API would serve {newest}. There is no "
              f"{METRICS} here to compare it with.")
        print(ask)
    elif newest == ours:
        print(f"A restarted API serves {ours}, the model in {METRICS}.")
        print(ask)
        print("It reads that model's scores on every request, so if that "
              f"names\n{ours}, the scores loaded under it are live, with no "
              "restart.\nIf it names another model, restart the API. If it "
              "still does after that,\nMODEL_VERSION is pinning it: set "
              f"MODEL_VERSION={ours} where\nthe API runs, or remove it, "
              "and restart again.")
    else:
        print(f"** A restarted API would serve {newest}, NOT {ours}, the "
              f"model in\n** {METRICS}: {newest} has the newer trained_at.")
        print(f"** Fix without touching data: set MODEL_VERSION={ours} "
              "where the API runs,\n** and restart it. That setting exists "
              "for exactly this (api/config.py on main).")
        print(f"** {ask}")


if __name__ == "__main__":
    main()
