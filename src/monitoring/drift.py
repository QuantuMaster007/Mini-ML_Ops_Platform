def zscore_shift(current_mean: float, baseline_mean: float, baseline_std: float) -> float:
    if baseline_std <= 1e-12:
        return 0.0
    return float((current_mean - baseline_mean) / baseline_std)

def abs_mean_shift_score(current_stats: dict, baseline_stats: dict) -> float:
    feats = baseline_stats["features"]
    scores = []
    for f in feats:
        s = zscore_shift(
            current_stats["feature_mean"][f],
            baseline_stats["feature_mean"][f],
            baseline_stats["feature_std"][f],
        )
        scores.append(abs(s))
    return float(max(scores)) if scores else 0.0

def abs_pred_mean_shift(current_pred_mean: float, baseline_pred_mean: float) -> float:
    return float(abs(current_pred_mean - baseline_pred_mean))
