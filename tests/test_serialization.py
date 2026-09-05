from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

JOBLIB_PATH = ROOT / "models" / "lasso_model.joblib"
ONNX_PATH = ROOT / "models" / "lasso_model.onnx"


def test_pickle_and_onnx_predictions_have_parity() -> None:
    """Pickle and ONNX should produce numerically equivalent predictions."""
    artifact = joblib.load(JOBLIB_PATH)

    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])

    X_df = pd.DataFrame(
        np.ones((1, len(feature_columns)), dtype=np.float64),
        columns=feature_columns,
    )

    pickle_prediction = float(np.asarray(model.predict(X_df)).reshape(-1)[0])

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_prediction = float(
        session.run(
            None,
            {
                input_name: X_df.to_numpy(
                    dtype=np.float32,
                )
            },
        )[0].reshape(-1)[0]
    )

    np.testing.assert_allclose(
        onnx_prediction,
        pickle_prediction,
        rtol=1e-5,
        atol=1e-6,
    )
