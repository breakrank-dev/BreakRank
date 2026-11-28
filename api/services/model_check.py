"""Startup check for the MODEL_VERSION override.

Without this check, a mistyped MODEL_VERSION on Render starts fine and quietly
ranks every list by usage instead of by the model, and nobody notices until a
demo. With it, the deploy fails and the log says exactly why.
"""

import os

from sqlalchemy import text

from api.db import engine


def check_model_override() -> None:
    """If MODEL_VERSION is set, refuse to start unless scores exist under that exact name."""
    override = os.environ.get("MODEL_VERSION", "").strip()
    if not override:
        return  # no override: the API serves the newest model, as before

    with engine.connect() as conn:
        found = conn.execute(
            text("SELECT 1 FROM prediction WHERE model_version = :v LIMIT 1"),
            {"v": override},
        ).first()

    if found is None:
        raise RuntimeError(
            f"MODEL_VERSION is set to {override!r}, but no scores are stored under that name. "
            "Fix the spelling (it must match model_run exactly), or delete the variable "
            "to serve the newest model."
        )
