from __future__ import annotations

import importlib

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeCleaner:
    """Capture the raw data received by preprocessing."""

    def __init__(self) -> None:
        self.last_input = None

    def transform(self, df):
        self.last_input = df.copy()
        return df.copy()


class FakeEngineer:
    """Pass data through unchanged."""

    def transform(self, df):
        return df.copy()


class FakeModel:
    """Return a deterministic prediction for API tests."""

    def predict(self, X):
        assert list(X.columns) == ["feature_a", "feature_b"]
        return np.array([np.log1p(250000.0)])


def create_artifact(path, cleaner):
    """Create a minimal test artifact."""
    joblib.dump(
        {
            "model": FakeModel(),
            "cleaner": cleaner,
            "engineer": FakeEngineer(),
            "feature_columns": ["feature_a", "feature_b"],
            "required_input_columns": ["feature_a", "feature_b"],
        },
        path,
    )


def test_health_without_model(monkeypatch, tmp_path):
    """Health should report model_loaded=False when artifact is absent."""
    monkeypatch.setenv(
        "MODEL_PATH",
        str(tmp_path / "missing.joblib"),
    )

    from house_price_mlops import api

    importlib.reload(api)

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict(monkeypatch, tmp_path):
    """Normal prediction should succeed."""
    artifact_path = tmp_path / "artifact.joblib"

    create_artifact(
        artifact_path,
        FakeCleaner(),
    )

    monkeypatch.setenv(
        "MODEL_PATH",
        str(artifact_path),
    )

    from house_price_mlops import api

    importlib.reload(api)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "feature_a": 1.0,
                    "feature_b": 2.0,
                }
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] == pytest.approx(250000.0)
    assert body["prediction_log"] == pytest.approx(
        np.log1p(250000.0)
    )


@pytest.mark.parametrize(
    "missing_value",
    [
        "NA",
        "N/A",
        "",
    ],
)
def test_predict_treats_missing_tokens_as_none(
    monkeypatch,
    tmp_path,
    missing_value,
):
    """Common missing-value strings should reach preprocessing as None."""
    artifact_path = tmp_path / "artifact.joblib"

    create_artifact(
        artifact_path,
        FakeCleaner(),
    )

    monkeypatch.setenv(
        "MODEL_PATH",
        str(artifact_path),
    )

    from house_price_mlops import api

    importlib.reload(api)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "feature_a": missing_value,
                    "feature_b": 2.0,
                }
            },
        )

    assert response.status_code == 200
    assert api._cleaner.last_input.loc[0, "feature_a"] is None


def test_predict_treats_explicit_null_as_none(
    monkeypatch,
    tmp_path,
):
    """Explicit JSON null should reach preprocessing as None."""
    artifact_path = tmp_path / "artifact.joblib"

    create_artifact(
        artifact_path,
        FakeCleaner(),
    )

    monkeypatch.setenv(
        "MODEL_PATH",
        str(artifact_path),
    )

    from house_price_mlops import api

    importlib.reload(api)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "feature_a": None,
                    "feature_b": 2.0,
                }
            },
        )

    assert response.status_code == 200
    assert api._cleaner.last_input.loc[0, "feature_a"] is None


def test_predict_treats_omitted_feature_as_none(
    monkeypatch,
    tmp_path,
):
    """An omitted feature should be treated as missing."""
    artifact_path = tmp_path / "artifact.joblib"

    create_artifact(
        artifact_path,
        FakeCleaner(),
    )

    monkeypatch.setenv(
        "MODEL_PATH",
        str(artifact_path),
    )

    from house_price_mlops import api

    importlib.reload(api)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "feature_b": 2.0,
                }
            },
        )

    assert response.status_code == 200
    assert api._cleaner.last_input.loc[0, "feature_a"] is None


def test_predict_preserves_real_values(
    monkeypatch,
    tmp_path,
):
    """Ordinary values must not be converted to missing values."""
    artifact_path = tmp_path / "artifact.joblib"

    create_artifact(
        artifact_path,
        FakeCleaner(),
    )

    monkeypatch.setenv(
        "MODEL_PATH",
        str(artifact_path),
    )

    from house_price_mlops import api

    importlib.reload(api)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "feature_a": "CollgCr",
                    "feature_b": 2.0,
                }
            },
        )

    assert response.status_code == 200
    assert api._cleaner.last_input.loc[0, "feature_a"] == "CollgCr"