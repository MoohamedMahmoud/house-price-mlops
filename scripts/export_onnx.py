from __future__ import annotations

from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "lasso_model.joblib"
ONNX_PATH = ROOT / "models" / "lasso_model.onnx"


def main() -> None:
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])

    initial_type = [
        (
            "input",
            FloatTensorType([None, len(feature_columns)]),
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17,
    )

    ONNX_PATH.write_bytes(onnx_model.SerializeToString())

    print(f"ONNX model saved to: {ONNX_PATH}")
    print(f"Model type: {type(model).__name__}")
    print(f"Feature count: {len(feature_columns)}")


if __name__ == "__main__":
    main()
