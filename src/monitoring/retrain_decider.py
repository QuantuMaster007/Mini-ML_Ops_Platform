import json
from pathlib import Path

import pandas as pd
import yaml

from src.monitoring.drift import abs_mean_shift_score, abs_pred_mean_shift

def main():
    cfg = yaml.safe_load(Path("configs/thresholds.yaml").read_text())
    t = cfg["retrain"]

    baseline_path = Path("artifacts/baseline_stats.json")
    if not baseline_path.exists():
        raise FileNotFoundError("artifacts/baseline_stats.json not found. Train first.")

    baseline = json.loads(baseline_path.read_text())

    df = pd.read_parquet("data/processed/dataset.parquet")
    recent = df.sample(n=min(len(df), 500), random_state=7)

    current_stats = {
        "feature_mean": {f: float(recent[f].mean()) for f in baseline["features"]},
        "pred_mean": float(recent["y"].mean()),
        "requests": int(len(recent)),
    }

    mean_shift = abs_mean_shift_score(current_stats, baseline)
    pred_shift = abs_pred_mean_shift(current_stats["pred_mean"], baseline["baseline_pred_mean"])

    retrain_needed = (
        current_stats["requests"] >= int(t["min_requests"]) and
        (mean_shift >= float(t["max_abs_feature_mean_shift"]) or pred_shift >= float(t["max_abs_pred_mean_shift"]))
    )

    out = {
        "requests": current_stats["requests"],
        "abs_feature_mean_shift": mean_shift,
        "abs_pred_mean_shift": pred_shift,
        "thresholds": t,
        "retrain_needed": bool(retrain_needed),
    }

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/retrain_decision.json").write_text(json.dumps(out, indent=2))
    print("✅ Wrote artifacts/retrain_decision.json")
    print(out)

if __name__ == "__main__":
    main()
