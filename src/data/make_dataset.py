import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.hashing import sha256_file

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    rng = np.random.default_rng(42)

    n = 5000
    age = rng.integers(18, 70, size=n)
    income = rng.normal(80000, 25000, size=n).clip(15000, 250000)

    # Simple deterministic label
    y = ((income + age * 300) > 90000).astype(int)

    df = pd.DataFrame({"age": age, "income": income.astype(float), "y": y})

    data_path = OUT / "dataset.parquet"
    df.to_parquet(data_path, index=False)

    manifest = {
        "dataset": str(data_path),
        "rows": int(len(df)),
        "schema": {c: str(df[c].dtype) for c in df.columns},
        "sha256": sha256_file(str(data_path)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "synthetic demo dataset v1"
    }

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("✅ Wrote:", data_path)
    print("✅ Manifest:", OUT / "manifest.json")

if __name__ == "__main__":
    main()
