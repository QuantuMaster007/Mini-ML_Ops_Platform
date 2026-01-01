# mini-mlops-platform (End-to-End, No Hype)

This repo demonstrates an end-to-end ML lifecycle:
**data versioning → training → model registry → containerized serving → monitoring → retraining triggers**

No fancy model. The point is production thinking.

## Stack
- PyTorch (tiny model)
- MLflow (tracking + registry)
- FastAPI (inference)
- Docker Compose (local platform)
- GitHub Actions (CI + scheduled retrain check)

## Quickstart (local)
```bash
make venv
source .venv/bin/activate
make install

docker compose down -v
make serve
export MLFLOW_TRACKING_URI="http://localhost:5050"

make data
make train
make promote
```

## Test inference
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":40,"income":90000}'
curl http://localhost:8000/metrics
```

## Monitoring + retraining triggers
We keep baseline stats from training (mean/std per feature + baseline pred mean).
A retrain can trigger when:
- enough requests exist (`min_requests`)
- feature mean shift (z-score) exceeds threshold (data drift proxy)
- prediction mean shift exceeds threshold (model behavior drift proxy)

Run:
```bash
python -m src.monitoring.retrain_decider
cat artifacts/retrain_decision.json
```

## Latency vs throughput tradeoff (what to say in interviews)
- Latency: single request time (FastAPI path)
- Throughput: requests/sec (batching, async, model loading strategy)
- This demo uses simple synchronous inference for clarity.
