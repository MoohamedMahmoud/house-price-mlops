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
            "required_input_columns": [
                "feature_a",
                "feature_b",
            ],
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
    assert body["prediction_log"] == pytest.approx(np.log1p(250000.0))


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


def test_metadata(monkeypatch, tmp_path):
    """Metadata endpoint should describe the loaded artifact."""
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
        response = client.get("/metadata")

    assert response.status_code == 200

    body = response.json()

    assert body["model_loaded"] is True
    assert body["model_type"] == "FakeModel"
    assert body["feature_count"] == 2
    assert body["required_input_count"] == 2
    assert body["feature_columns"] == [
        "feature_a",
        "feature_b",
    ]
    assert body["required_input_columns"] == [
        "feature_a",
        "feature_b",
    ]


def test_predict_batch(monkeypatch, tmp_path):
    """Batch prediction should return one prediction per instance."""
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
            "/predict/batch",
            json={
                "instances": [
                    {
                        "features": {
                            "feature_a": 1.0,
                            "feature_b": 2.0,
                        }
                    },
                    {
                        "features": {
                            "feature_a": 3.0,
                            "feature_b": 4.0,
                        }
                    },
                ]
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert len(body["predictions"]) == 2

    for prediction in body["predictions"]:
        assert prediction["prediction"] == pytest.approx(250000.0)
        assert prediction["prediction_log"] == pytest.approx(np.log1p(250000.0))


def test_correlation_id_is_returned(
    monkeypatch,
    tmp_path,
):
    """Provided correlation ID should be returned in the response."""
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

    correlation_id = "test-correlation-id"

    with TestClient(api.app) as client:
        response = client.get(
            "/health",
            headers={
                "X-Correlation-ID": correlation_id,
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == correlation_id


def test_correlation_id_is_generated(
    monkeypatch,
    tmp_path,
):
    """API should generate a correlation ID when none is provided."""
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
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"]
