import os
import time
import numpy as np
import mlflow
from fastapi import FastAPI

from src.serving.schemas import PredictRequest, PredictResponse
from src.serving.load_model import load_model

app = FastAPI()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(mlflow_uri)

MODEL, MODEL_URI = load_model()
STATS = {"requests": 0, "pred_sum": 0.0, "latencies_ms": []}

@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.get("/metrics")
def metrics():
    req = STATS["requests"]
    pred_mean = (STATS["pred_sum"] / req) if req else 0.0
    p95 = float(np.percentile(STATS["latencies_ms"], 95)) if STATS["latencies_ms"] else 0.0
    return {"requests": req, "pred_mean": pred_mean, "latency_p95_ms": p95}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    x = np.array([[req.age, req.income]], dtype=float)
    y = float(MODEL.predict(x)[0])
    latency_ms = (time.perf_counter() - start) * 1000.0

    STATS["requests"] += 1
    STATS["pred_sum"] += y
    STATS["latencies_ms"].append(latency_ms)

    return PredictResponse(prediction=y, latency_ms=round(latency_ms, 3), model_uri=MODEL_URI)
