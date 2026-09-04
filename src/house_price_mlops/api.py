from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from house_price_mlops.features import AmesFeatureEngineer
from house_price_mlops.preprocessing import AmesCleaner

LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/lasso_model.joblib"))

MISSING_VALUE_TOKENS = {"", "NA", "N/A", "NULL"}


app = FastAPI(
    title="Ames House Price API",
    version="1.0.0",
    description="Predict SalePrice for the Ames House Prices dataset.",
)


class PredictRequest(BaseModel):
    """Raw Ames house features for one prediction."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Raw Ames feature names/values. "
            "Missing values may be represented by null, NA, N/A, "
            "an empty string, or by omitting the field. "
            "Id and SalePrice are not required."
        ),
    )


class PredictResponse(BaseModel):
    prediction: float
    prediction_log: float


_model: Any | None = None
_cleaner: AmesCleaner | None = None
_engineer: AmesFeatureEngineer | None = None
_feature_columns: list[str] = []
_required_input_columns: list[str] = []


def normalize_missing_value(value: Any) -> Any:
    """Convert common API representations of missing values to None."""
    if isinstance(value, str) and value.strip().upper() in MISSING_VALUE_TOKENS:
        return None

    return value


def load_artifact() -> None:
    """Load the trained model and preprocessing objects."""
    global _model, _cleaner, _engineer
    global _feature_columns, _required_input_columns

    if not MODEL_PATH.exists():
        LOGGER.warning("Model artifact not found: %s", MODEL_PATH)
        return

    artifact = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "cleaner",
        "engineer",
        "feature_columns",
        "required_input_columns",
    }

    missing = required_keys.difference(artifact)

    if missing:
        raise RuntimeError(f"Invalid model artifact; missing keys: {sorted(missing)}")

    _model = artifact["model"]
    _cleaner = artifact["cleaner"]
    _engineer = artifact["engineer"]
    _feature_columns = list(artifact["feature_columns"])
    _required_input_columns = list(artifact["required_input_columns"])

    LOGGER.info(
        "Model artifact loaded: model=%s features=%d",
        type(_model).__name__,
        len(_feature_columns),
    )


@app.on_event("startup")
def startup_event() -> None:
    """Load the model artifact when the API starts."""
    load_artifact()


@app.get("/health")
def health() -> dict[str, Any]:
    """Return API and model health information."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model": "Lasso" if _model is not None else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict house price from raw Ames features."""
    if _model is None or _cleaner is None or _engineer is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded",
        )

    try:
        # Build a complete raw input row.
        #
        # A field may be:
        #   - explicitly null
        #   - "NA"
        #   - "N/A"
        #   - ""
        #   - completely omitted
        #
        # All of these become None.
        #
        # The API does NOT decide how to impute the value.
        # AmesCleaner remains responsible for preprocessing.
        raw_features = {
            column: normalize_missing_value(request.features.get(column))
            for column in _required_input_columns
        }

        raw_df = pd.DataFrame([raw_features])

        clean_df = _cleaner.transform(raw_df)

        encoded_df = _engineer.transform(clean_df)

        # Preserve the exact feature order used during training.
        X = encoded_df.reindex(
            columns=_feature_columns,
            fill_value=0,
        )

        prediction_log = float(np.asarray(_model.predict(X)).reshape(-1)[0])

        prediction = float(np.expm1(prediction_log))

        return PredictResponse(
            prediction=prediction,
            prediction_log=prediction_log,
        )

    except (KeyError, ValueError, TypeError) as exc:
        LOGGER.exception("Prediction preprocessing failed")

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
