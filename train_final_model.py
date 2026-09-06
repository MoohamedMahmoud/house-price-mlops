from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from house_price_mlops.pipeline import build_dataset
from house_price_mlops.train import build_tuned_model

RAW_TRAIN_PATH = Path("data/raw/train.csv")
MODEL_PATH = Path("models/lasso_model.joblib")


FINAL_MODEL_NAME = "Lasso"
FINAL_MODEL_PARAMS = {"model__alpha": 0.003}
KEEP_OUTLIERS = True


def main() -> None:
    if not RAW_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {RAW_TRAIN_PATH}")

    raw_train = pd.read_csv(RAW_TRAIN_PATH)

    X, y, cleaner, engineer = build_dataset(
        raw_train,
        keep_outliers=KEEP_OUTLIERS,
        fit=True,
    )

    model = build_tuned_model(FINAL_MODEL_NAME, FINAL_MODEL_PARAMS)
    model.fit(X, y)

    # The API needs both the fitted model and the fitted preprocessing objects.
    # required_input_columns excludes Id and SalePrice because the API predicts
    # SalePrice and the cleaner removes Id itself.
    required_input_columns = [
        c for c in raw_train.columns if c not in {"Id", "SalePrice"}
    ]

    artifact = {
        "model": model,
        "cleaner": cleaner,
        "engineer": engineer,
        "feature_columns": list(X.columns),
        "required_input_columns": required_input_columns,
        "model_name": FINAL_MODEL_NAME,
        "model_params": FINAL_MODEL_PARAMS,
        "keep_outliers": KEEP_OUTLIERS,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved final model artifact: {MODEL_PATH}")
    print(f"Model: {FINAL_MODEL_NAME}")
    print(f"Parameters: {FINAL_MODEL_PARAMS}")
    print(f"Training rows: {len(X)}")
    print(f"Features: {X.shape[1]}")


if __name__ == "__main__":
    main()
