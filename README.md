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
## Smoke test evidence (local)

Outputs saved under `docs/smoke/`:

- `health.txt`
- `predict.txt`
- `metrics.txt`
- `docs.txt` (Swagger UI HTML response)

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

## Latency vs throughput tradeoff
- Latency: single request time (FastAPI path)
- Throughput: requests/sec (batching, async, model loading strategy)
- This demo uses simple synchronous inference for clarity.


---

## 🤝 Contributing

This is a demonstration project for portfolio. If you'd like to extend it:

1. Fork the repository
2. Create a feature branch
3. Add enhancements (new models, visualizations, data sources)
4. Submit a pull request

---

## 📧 Contact

Let's connect! Whether you have a question or just want to say hi, feel free to reach out.

| Platform | Link |
| :--- | :--- |
| **👤 Name** | Sourabh Tarodekar |
| **✉️ Email** | [sourabh232@gmail.com](mailto:sourabh232@gmail.com) |
| **💼 LinkedIn** | [linkedin.com/in/sourabh232](https://www.linkedin.com/in/sourabh232) |
| **🚀 Portfolio** | [QuantuMaster007 Portfolio](https://github.com/QuantuMaster007/sourabh232.git) |

---

## 📄 License

MIT License - See LICENSE file for details

---
