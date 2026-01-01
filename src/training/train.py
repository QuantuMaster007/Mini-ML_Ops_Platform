import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.training.model import MLP

DATA_DIR = Path("data/processed")

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("mini-mlops-platform")

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json not found. Run: python -m src.data.make_dataset")

    manifest = json.loads(manifest_path.read_text())
    df = pd.read_parquet(DATA_DIR / "dataset.parquet")

    features = ["age", "income"]
    target = "y"

    X = df[features].values.astype("float32")
    y = df[target].values.astype("float32").reshape(-1, 1)

    set_seed(42)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva)
    yva_t = torch.tensor(yva)

    model = MLP(hidden_dim=32)
    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    batch_size = 256

    def batches(Xt, yt, bs):
        n = Xt.shape[0]
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            j = idx[i:i+bs]
            yield Xt[j], yt[j]

    Path("artifacts").mkdir(exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "pytorch_mlp")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", 1e-3)

        for epoch in range(epochs):
            model.train()
            losses = []
            for xb, yb in batches(Xtr_t, ytr_t, batch_size):
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            mlflow.log_metric("train_loss", float(np.mean(losses)), step=epoch)

        model.eval()
        with torch.no_grad():
            va_prob = model(Xva_t).numpy().reshape(-1)
            va_pred = (va_prob > 0.5).astype(int)
            val_acc = accuracy_score(yva.reshape(-1).astype(int), va_pred)

        mlflow.log_metric("val_accuracy", float(val_acc))

        baseline = {
            "data_sha256": manifest["sha256"],
            "features": features,
            "feature_mean": {f: float(df[f].mean()) for f in features},
            "feature_std": {f: float(df[f].std()) for f in features},
            "baseline_pred_mean": float(va_prob.mean())
        }
        mlflow.log_dict(manifest, "data_manifest.json")
        mlflow.log_dict(baseline, "baseline_stats.json")

        reg_name = "mini-mlops-platform-model"
        info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=reg_name
        )

        metrics_out = {
            "run_id": run.info.run_id,
            "val_accuracy": float(val_acc),
            "registered_model_name": reg_name,
            "model_uri": info.model_uri
        }
        (Path("artifacts") / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

        print("✅ run_id:", run.info.run_id)
        print("✅ val_accuracy:", val_acc)
        print("✅ registered:", reg_name)

if __name__ == "__main__":
    main()
