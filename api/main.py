from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import config
from api.routes import packages


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.MODEL_VERSION, config.POSITIVE_RATE = config.resolve_model()
    print(f"Serving model: {config.MODEL_VERSION} (floor {config.POSITIVE_RATE})")
    yield


app = FastAPI(title="BreakRank API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(packages.router)


@app.get("/health")
def health():
    return {"ok": True, "model_version": config.MODEL_VERSION}