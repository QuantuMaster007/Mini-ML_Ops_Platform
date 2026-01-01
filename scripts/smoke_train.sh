#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5050}

python -m src.data.make_dataset
python -m src.training.train
python -m src.registry.promote --min-acc 0.80
echo "✅ smoke_train ok"
