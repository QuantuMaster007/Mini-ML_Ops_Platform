import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="mini-mlops-platform-model")
    ap.add_argument("--min-acc", type=float, default=0.80)
    ap.add_argument("--metrics-file", default="artifacts/metrics.json")
    args = ap.parse_args()

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(mlflow_uri)

    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json not found. Run training first.")

    m = json.loads(metrics_path.read_text())
    run_id = m["run_id"]
    val_acc = float(m["val_accuracy"])

    if val_acc < args.min_acc:
        raise RuntimeError(f"Quality gate failed: val_accuracy {val_acc:.4f} < {args.min_acc:.4f}")

    client = MlflowClient()

    versions = client.search_model_versions(f"name='{args.model_name}'")
    target_version = None
    for v in versions:
        if v.run_id == run_id:
            target_version = v.version
            break

    if target_version is None:
        raise RuntimeError("Could not find a registered model version for this run.")

    client.transition_model_version_stage(
        name=args.model_name,
        version=str(target_version),
        stage="Production",
        archive_existing_versions=True
    )

    try:
        client.set_registered_model_alias(args.model_name, "production", str(target_version))
    except Exception:
        pass

    print(f"✅ Promoted {args.model_name} v{target_version} -> Production (val_accuracy={val_acc:.4f})")

if __name__ == "__main__":
    main()
