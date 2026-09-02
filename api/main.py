from fastapi import FastAPI

app = FastAPI(title="BreakRank API")


@app.get("/health")
def health():
    return {"ok": True, "model_version": "v0"}