from pydantic import BaseModel, Field


class Breakage(BaseModel):
    symbol: str
    kind: str
    sub_target: str = Field(
        "", description="Parameter name for PARAMETER_* kinds, else empty."
    )
    score: float | None = Field(
        None, description="Null when no model has scored this breakage."
    )
    user_count: int
    inherited_by: int = Field(
        0, description="How many other classes inherit this change."
    )
    explanation: str
    is_private: bool


class BreakagesResponse(BaseModel):
    package: str
    version: str
    model_version: str | None
    ranking: str = Field(description='"model" or "usage_fallback"')
    analysis_status: str = Field(
        description="analysed | analysed_clean | analysis_failed | "
                    "no_source | no_baseline | yanked"
    )
    total: int
    breakages: list[Breakage]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail