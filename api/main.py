from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import config
from api.routes import analyze, packages
from api.services.model_check import check_model_override


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_model_override()
    config.MODEL_VERSION, config.POSITIVE_RATE = config.resolve_model()
    print(f"Serving model: {config.MODEL_VERSION} (floor {config.POSITIVE_RATE})")
    yield


app = FastAPI(title="BreakRank API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://break-rank.vercel.app", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Every error has one shape: {"error": {"code": ..., "message": ...}}.
# FastAPI's own HTTPException is a subclass of Starlette's, so this
# catches both ours (e.g. not_tracked) and the built-in 404/405.
@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code", "http_error")
        message = exc.detail.get("message", "")
    else:
        code, message = "http_error", str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": message}},
        headers=getattr(exc, "headers", None),
    )


# Bad input (e.g. limit=0, or a requirements text that's too long).
@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    message = first.get("msg", "The request was not valid.")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": f"{where}: {message}" if where else message,
            }
        },
    )


app.include_router(packages.router)
app.include_router(analyze.router)


@app.get("/health")
def health():
    return {"ok": True, "model_version": config.MODEL_VERSION}