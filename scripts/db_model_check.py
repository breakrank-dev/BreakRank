"""
Which model will the site serve?

    python scripts/db_model_check.py

READ ONLY. Every statement in this file is a SELECT, so it is safe to run
against the live database at any time.

WHY THIS EXISTS. The API does not ask for the newest model on every
request. It picks one ONCE, when it starts: the model_run row with the
newest trained_at, unless the MODEL_VERSION setting pins one (decision 8,
api/config.py on main). So two things can leave the site on an old model
after `ml/db.py --scores` has loaded a new one:

  1. The API has not restarted since the load. It keeps serving what it
     picked when it last started.
  2. The version was already in the table from an earlier load. The
     loader never moves trained_at on a re-load (NOTES §13.5, so that
     re-scoring an old model cannot make it current by accident), and so
     a re-loaded version keeps its old date and may not be the newest.

This prints every model_run row, newest first, with how many predictions
each holds, and says which one a restarted API would serve and whether
that is the model in artifacts/metrics.json.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import connect  # noqa: E402

METRICS = pathlib.Path("artifacts") / "metrics.json"


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

    print(f"\nmodel_run rows, newest first ({breakages:,} breakage rows in "
          "the table):\n")
    print(f"  {'version':<34}{'trained_at':<28}{'pr_auc':>8}"
          f"{'predictions':>13}")
    for version, at, pr, n in runs:
        mark = "   <- artifacts/metrics.json" if version == ours else ""
        print(f"  {version:<34}{str(at)[:26]:<28}"
              f"{'-' if pr is None else f'{pr:.4f}':>8}{n:>13,}{mark}")

    if not runs:
        print("  (none) Load one with: python ml/db.py --scores")
        return
    newest = runs[0][0]
    print()
    if ours is None:
        print(f"A restarted API would serve {newest}. There is no "
              f"{METRICS} here to compare it with.")
    elif newest == ours:
        print(f"A restarted API serves {ours}, the model in {METRICS}.")
        print("The API picks its model when it starts, so if it has not "
              "restarted since\nthe load, it is still serving whatever it "
              "picked then. Ask whoever runs it\nto restart it, or to set "
              f"MODEL_VERSION={ours}.")
    else:
        print(f"** A restarted API would serve {newest}, NOT {ours}, the "
              f"model in\n** {METRICS}: {newest} has the newer trained_at.")
        print(f"** Fix without touching data: set MODEL_VERSION={ours} "
              "where the API runs,\n** and restart it. That setting exists "
              "for exactly this (api/config.py on main).")


if __name__ == "__main__":
    main()
