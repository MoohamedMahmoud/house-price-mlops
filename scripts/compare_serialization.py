from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

JOBLIB_PATH = ROOT / "models" / "lasso_model.joblib"
ONNX_PATH = ROOT / "models" / "lasso_model.onnx"

N_RUNS = 1000


def main() -> None:
    artifact = joblib.load(JOBLIB_PATH)

    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])

    # Build one deterministic input with the same feature names
    # used when the sklearn pipeline was fitted.
    X_df = pd.DataFrame(
        np.ones((1, len(feature_columns)), dtype=np.float64),
        columns=feature_columns,
    )

    # ONNX Runtime expects a NumPy array.
    X_onnx = X_df.to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # Pickle / joblib
    # ------------------------------------------------------------------
    pickle_prediction = float(np.asarray(model.predict(X_df)).reshape(-1)[0])

    pickle_start = perf_counter()

    for _ in range(N_RUNS):
        model.predict(X_df)

    pickle_elapsed = perf_counter() - pickle_start
    pickle_latency_ms = (pickle_elapsed / N_RUNS) * 1000

    # ------------------------------------------------------------------
    # ONNX Runtime
    # ------------------------------------------------------------------
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_prediction = float(
        session.run(
            None,
            {input_name: X_onnx},
        )[0].reshape(-1)[0]
    )

    onnx_start = perf_counter()

    for _ in range(N_RUNS):
        session.run(
            None,
            {input_name: X_onnx},
        )

    onnx_elapsed = perf_counter() - onnx_start
    onnx_latency_ms = (onnx_elapsed / N_RUNS) * 1000

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    absolute_difference = abs(pickle_prediction - onnx_prediction)

    speedup = pickle_latency_ms / onnx_latency_ms

    print("=== Serialization Comparison ===")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Runs: {N_RUNS}")
    print()
    print(f"Pickle prediction:   {pickle_prediction:.12f}")
    print(f"ONNX prediction:     {onnx_prediction:.12f}")
    print(f"Absolute difference: {absolute_difference:.12e}")
    print()
    print(f"Pickle latency: {pickle_latency_ms:.6f} ms")
    print(f"ONNX latency:   {onnx_latency_ms:.6f} ms")
    print(f"ONNX speedup:   {speedup:.2f}x")


if __name__ == "__main__":
    main()
