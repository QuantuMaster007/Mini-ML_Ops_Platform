#!/usr/bin/env bash
set -euo pipefail
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":40,"income":90000}'
curl -s http://localhost:8000/metrics
echo "✅ smoke_infer ok"
