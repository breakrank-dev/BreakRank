"""Request and response shapes for POST /analyze."""

from pydantic import BaseModel, Field

from api.schemas.breakages import Breakage

MAX_CHARS = 20_000
PER_RELEASE_DEFAULT = 5
PER_RELEASE_MAX = 20


class AnalyzeRequest(BaseModel):
    requirements: str = Field(..., max_length=MAX_CHARS)
    per_release: int = Field(PER_RELEASE_DEFAULT, ge=1, le=PER_RELEASE_MAX)


class ReleaseResult(BaseModel):
    version: str
    analysis_status: str
    ranking: str  # "model" or "usage_fallback", as on the release endpoint
    total: int  # public changes in this release, after keeping only the newest copy of each
    n_imported: int  # how many of those any of the scanned packages import
    breakages: list[Breakage]  # the top `per_release`, ranked within this release


class PackageResult(BaseModel):
    package: str
    current: str
    latest: str
    releases: list[ReleaseResult]  # newest first
    older_releases_not_shown: int


class UpToDate(BaseModel):
    package: str
    version: str
    latest: str


class SkippedLine(BaseModel):
    line_number: int
    line: str
    reason: str


class AnalyzeResponse(BaseModel):
    model_version: str | None
    packages: list[PackageResult]
    up_to_date: list[UpToDate]
    not_tracked: list[str]
    skipped: list[SkippedLine]
