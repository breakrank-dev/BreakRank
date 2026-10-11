"""POST /analyze: paste a requirements.txt, get the breaking changes between each
pinned version and the newest release we hold.

Results are grouped by package, then by release. Within a release, changes are
ranked by the model exactly as on the release endpoint. Releases are never
merged into one list, because a score only orders changes within one release:
it is not comparable across releases or packages.
"""

from collections import defaultdict

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from api import config
from api.db import engine
from api.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
    PackageResult,
    ReleaseResult,
    SkippedLine,
    UpToDate,
)
from api.schemas.breakages import Breakage
from api.services.explanation import render
from api.services.requirements import parse_requirements, releases_after

router = APIRouter()

MAX_PACKAGES = 200
MAX_RELEASES_PER_PACKAGE = 10  # the newest ones; the rest are counted, not shown

# Match package names the way pip does: case-insensitive, and - _ . all equal.
PACKAGES_SQL = text("""
    SELECT id, name, lower(regexp_replace(name, '[-_.]+', '-', 'g')) AS canon
    FROM package
    WHERE lower(regexp_replace(name, '[-_.]+', '-', 'g')) = ANY(CAST(:names AS TEXT[]))
""")

RELEASES_SQL = text("""
    SELECT id, package_id, version, analysis_status
    FROM release
    WHERE package_id = ANY(CAST(:package_ids AS BIGINT[]))
""")

# :release_ids lists each package's releases oldest first, so a higher
# array position means a newer release of that package.
BREAKAGES_SQL = text("""
    WITH in_range AS (
        SELECT b.id, b.release_id, r.package_id, b.symbol_path, b.kind, b.sub_target,
               b.detail, COALESCE(b.inherited_by, 0) AS inherited_by,
               array_position(CAST(:release_ids AS BIGINT[]), CAST(b.release_id AS BIGINT)) AS pos
        FROM breakage b
        JOIN release r ON r.id = b.release_id
        WHERE b.release_id = ANY(CAST(:release_ids AS BIGINT[]))
          AND b.is_private = FALSE
    ),
    newest_copy AS (
        -- The same change can appear in several releases in the range.
        -- Keep only the newest release's copy (decision 11).
        SELECT DISTINCT ON (package_id, symbol_path, kind, sub_target) *
        FROM in_range
        ORDER BY package_id, symbol_path, kind, sub_target, pos DESC
    ),
    ranked AS (
        SELECT n.release_id, n.symbol_path, n.kind, n.sub_target, n.detail, n.inherited_by,
               p.score,
               COALESCE(u.user_count, 0) AS user_count,
               COUNT(*) OVER per_release AS release_total,
               COUNT(p.score) OVER per_release AS release_scored,
               COUNT(*) FILTER (WHERE COALESCE(u.user_count, 0) > 0) OVER per_release
                   AS release_imported,
               ROW_NUMBER() OVER (
                   PARTITION BY n.release_id
                   ORDER BY p.score DESC NULLS LAST, COALESCE(u.user_count, 0) DESC,
                            n.symbol_path ASC, n.sub_target ASC
               ) AS rn
        FROM newest_copy n
        LEFT JOIN prediction p ON p.breakage_id = n.id AND p.model_version = :model_version
        LEFT JOIN usage_index u ON u.symbol_path = n.symbol_path
        WINDOW per_release AS (PARTITION BY n.release_id)
    )
    SELECT * FROM ranked
    WHERE rn <= :per_release
    ORDER BY release_id, rn
""")


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    pins, skipped = parse_requirements(body.requirements)
    if len(pins) > MAX_PACKAGES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "too_many_packages",
                "message": f"Paste at most {MAX_PACKAGES} pinned packages at a time.",
            },
        )

    with engine.connect() as conn:
        # 1. Which of the pinned packages do we track?
        packages = {}
        if pins:
            rows = conn.execute(PACKAGES_SQL, {"names": [p.name for p in pins]}).all()
            packages = {row.canon: row for row in rows}

        # 2. Every release we hold for those packages.
        releases_by_package = defaultdict(list)
        if packages:
            rows = conn.execute(
                RELEASES_SQL, {"package_ids": [row.id for row in packages.values()]}
            ).all()
            for row in rows:
                releases_by_package[row.package_id].append(row)

        # 3. For each pin, decide which releases to show.
        plans, up_to_date, not_tracked = [], [], []
        for pin in pins:
            package = packages.get(pin.name)
            if package is None:
                not_tracked.append(pin.name)
                continue
            latest, after = releases_after(pin.version, releases_by_package[package.id])
            if latest is None:
                not_tracked.append(pin.name)
            elif not after:
                up_to_date.append(
                    UpToDate(package=package.name, version=str(pin.version), latest=latest.raw)
                )
            else:
                shown = after[-MAX_RELEASES_PER_PACKAGE:]  # still oldest first
                plans.append((pin, package, latest, shown, len(after) - len(shown)))

        # 4. The ranked changes for every shown release, in one query.
        rows_by_release = defaultdict(list)
        release_ids = [r.id for _, _, _, shown, _ in plans for r in shown]
        if release_ids:
            rows = conn.execute(
                BREAKAGES_SQL,
                {
                    "release_ids": release_ids,
                    "model_version": config.MODEL_VERSION,
                    "per_release": body.per_release,
                },
            ).all()
            for row in rows:
                rows_by_release[row.release_id].append(row)

    # 5. Build the response: packages, then releases newest first.
    results = []
    for pin, package, latest, shown, hidden in plans:
        releases = [_release_result(r, rows_by_release[r.id]) for r in reversed(shown)]
        results.append(
            PackageResult(
                package=package.name,
                current=str(pin.version),
                latest=latest.raw,
                releases=releases,
                older_releases_not_shown=hidden,
            )
        )

    # Packages whose changes are imported most widely first. user_count is a
    # plain count, so unlike the score it can be compared across packages.
    results.sort(key=lambda p: (-_max_user_count(p), p.package))

    return AnalyzeResponse(
        model_version=config.MODEL_VERSION,
        packages=results,
        up_to_date=up_to_date,
        not_tracked=not_tracked,
        skipped=[SkippedLine(line_number=s.line_number, line=s.line, reason=s.reason) for s in skipped],
    )


def _release_result(release, rows) -> ReleaseResult:
    first = rows[0] if rows else None
    return ReleaseResult(
        version=release.raw,
        analysis_status=release.analysis_status,
        ranking="model" if first is not None and first.release_scored > 0 else "usage_fallback",
        total=first.release_total if first is not None else 0,
        n_imported=first.release_imported if first is not None else 0,
        breakages=[
            Breakage(
                symbol=row.symbol_path,
                kind=row.kind,
                sub_target=row.sub_target,
                score=row.score,
                user_count=row.user_count,
                inherited_by=row.inherited_by,
                explanation=render(
                    row.kind,
                    row.symbol_path,
                    row.user_count,
                    row.detail,
                    sub_target=row.sub_target,
                    inherited_by=row.inherited_by,
                ),
                is_private=False,
            )
            for row in rows
        ],
    )


def _max_user_count(package: PackageResult) -> int:
    return max((b.user_count for r in package.releases for b in r.breakages), default=0)
