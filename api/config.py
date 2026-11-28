"""Application settings and startup-time model resolution."""
import os

from sqlalchemy import text

from api.db import engine

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
MAX_ANALYZE_PACKAGES = 100


def resolve_model() -> tuple[str | None, float | None]:
    """Return (version, positive_rate) for the model this API serves.

    Decision 8: newest model_run by trained_at, resolved once at startup.
    ORDER BY version would be wrong — 'v0-fake' sorts after the real model
    name as text, so the fixture would win with 0.0 PR-AUC behind it.

    MODEL_VERSION pins a version so a bad retrain can be rolled back in
    thirty seconds without touching code or data.

    Decision 13: positive_rate travels with the version, because pr_auc must
    never be displayed without its floor.
    """
    pinned = os.getenv("MODEL_VERSION")

    with engine.connect() as conn:
        if pinned:
            row = conn.execute(
                text("SELECT version, positive_rate FROM model_run "
                     "WHERE version = :v"),
                {"v": pinned},
            ).first()
        else:
            row = conn.execute(
                text("SELECT version, positive_rate FROM model_run "
                     "ORDER BY trained_at DESC LIMIT 1")
            ).first()

    return (row[0], row[1]) if row else (None, None)


MODEL_VERSION: str | None = None
POSITIVE_RATE: float | None = None