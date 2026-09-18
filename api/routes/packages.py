from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from api import config
from api.db import engine
from api.schemas.breakages import Breakage, BreakagesResponse
from api.services import explanation

router = APIRouter(prefix="/packages", tags=["packages"])


RELEASE_SQL = """
SELECT r.analysis_status
FROM release r
JOIN package pk ON pk.id = r.package_id
WHERE pk.name = :name AND r.version = :version
"""

COUNT_SQL = """
SELECT count(*)
FROM breakage b
JOIN release r  ON r.id = b.release_id
JOIN package pk ON pk.id = r.package_id
WHERE pk.name = :name
  AND r.version = :version
  AND (:include_private OR b.is_private = FALSE)
"""

BREAKAGES_SQL = """
SELECT
    b.symbol_path,
    b.kind,
    b.sub_target,
    b.is_private,
    b.inherited_by,
    b.detail,
    COALESCE(u.user_count, 0) AS user_count,
    p.score
FROM breakage b
JOIN release r  ON r.id = b.release_id
JOIN package pk ON pk.id = r.package_id
LEFT JOIN prediction  p ON p.breakage_id = b.id
                       AND p.model_version = :model_version
LEFT JOIN usage_index u ON u.symbol_path = b.symbol_path
WHERE pk.name = :name
  AND r.version = :version
  AND (:include_private OR b.is_private = FALSE)
ORDER BY p.score DESC NULLS LAST,
         COALESCE(u.user_count, 0) DESC,
         b.symbol_path ASC,
         b.sub_target ASC
LIMIT :limit
"""


@router.get(
    "/{name}/releases/{version}/breakages",
    response_model=BreakagesResponse,
)
def get_breakages(
    name: str,
    version: str,
    limit: int = Query(config.DEFAULT_LIMIT, ge=1, le=config.MAX_LIMIT),
    include_private: bool = False,
):
    params = {
        "name": name,
        "version": version,
        "model_version": config.MODEL_VERSION,
        "include_private": include_private,
        "limit": limit,
    }

    with engine.connect() as conn:
        # Decision 1: "we don't have this" and "we have it and it's empty"
        # are different answers and must not both be an empty list.
        status = conn.execute(
            text(RELEASE_SQL), {"name": name, "version": version}
        ).scalar()

        if status is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "not_tracked",
                    "message": f"BreakRank has no record of {name}=={version}.",
                },
            )

        total = conn.execute(text(COUNT_SQL), params).scalar()
        rows = conn.execute(text(BREAKAGES_SQL), params).mappings().all()

    # Decision 2: if nothing is scored we ranked by usage, and we say so
    # rather than implying a model ordered this.
    has_scores = any(r["score"] is not None for r in rows)

    return BreakagesResponse(
        package=name,
        version=version,
        model_version=config.MODEL_VERSION,
        ranking="model" if has_scores else "usage_fallback",
        analysis_status=status,
        total=total,
        breakages=[
            Breakage(
                symbol=r["symbol_path"],
                kind=r["kind"],
                sub_target=r["sub_target"],
                score=r["score"],
                user_count=r["user_count"],
                inherited_by=r["inherited_by"],
                is_private=r["is_private"],
                explanation=explanation.render(
                    r["kind"],
                    r["symbol_path"],
                    r["user_count"],
                    r["detail"] or {},
                    r["sub_target"],
                    r["inherited_by"],
                ),
            )
            for r in rows
        ],
    )