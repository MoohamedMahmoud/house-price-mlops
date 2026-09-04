"""
Tests for src/evaluate.py (evaluate_model, compare_models).

Run with:
    uv run pytest tests/test_evaluate.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from house_price_mlops.evaluate import compare_models, evaluate_model


@pytest.fixture
def toy_data():
    """
    A tiny, perfectly linear relationship in log space, so a plain
    LinearRegression can fit it essentially exactly -- makes the expected
    metrics predictable enough to assert on directly, rather than needing
    to trust the function's own arithmetic.
    """
    X_train = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y_train = pd.Series(2.0 * X_train["x"] + 10.0)

    X_val = pd.DataFrame({"x": np.arange(20, 25, dtype=float)})
    y_val = pd.Series(2.0 * X_val["x"] + 10.0)

    return X_train, y_train, X_val, y_val


def test_evaluate_model_near_zero_error_on_perfect_fit(toy_data):
    X_train, y_train, X_val, y_val = toy_data

    result, _, _, _ = evaluate_model(
        LinearRegression(),
        "Linear Regression",
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert result["Train log-RMSE"] < 1e-8
    assert result["Validation log-RMSE"] < 1e-8


def test_evaluate_model_returns_expected_keys(toy_data):
    X_train, y_train, X_val, y_val = toy_data

    result, *_ = evaluate_model(
        LinearRegression(),
        "LR",
        X_train,
        y_train,
        X_val,
        y_val,
    )

    expected_keys = {
        "Model",
        "Train log-RMSE",
        "Validation log-RMSE",
        "Train RMSE",
        "Validation RMSE",
        "Train/Val Gap",
    }

    assert set(result.keys()) == expected_keys
    assert result["Model"] == "LR"


def test_gap_is_val_minus_train(toy_data):
    X_train, y_train, X_val, y_val = toy_data

    result, *_ = evaluate_model(
        LinearRegression(),
        "LR",
        X_train,
        y_train,
        X_val,
        y_val,
    )

    expected_gap = result["Validation log-RMSE"] - result["Train log-RMSE"]

    assert result["Train/Val Gap"] == pytest.approx(expected_gap)


def test_returned_model_is_fitted(toy_data):
    X_train, y_train, X_val, y_val = toy_data

    _, fitted_model, _, _ = evaluate_model(
        LinearRegression(),
        "LR",
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert hasattr(fitted_model, "coef_")


def test_residuals_are_actual_minus_predicted(toy_data):
    X_train, y_train, X_val, y_val = toy_data

    _, _, y_val_pred, residuals = evaluate_model(
        LinearRegression(),
        "LR",
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert np.allclose(
        residuals,
        y_val.values - y_val_pred,
        atol=1e-8,
    )


def test_rmse_in_dollars_uses_expm1():
    """
    Train RMSE ($) must be computed by inverting the log-scale predictions
    with expm1, not just re-reporting the log-scale error under a
    different label. Uses a fixture with some error.
    """
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y_train_log = pd.Series([10.0, 10.4, 10.9, 11.5])

    result, _, _, _ = evaluate_model(
        LinearRegression(),
        "LR",
        X_train,
        y_train_log,
        X_train,
        y_train_log,
    )

    manual_model = LinearRegression().fit(X_train, y_train_log)
    manual_pred_log = manual_model.predict(X_train)

    manual_rmse_dollars = np.sqrt(
        np.mean((np.expm1(y_train_log) - np.expm1(manual_pred_log)) ** 2)
    )

    assert result["Train RMSE"] == pytest.approx(manual_rmse_dollars)

    assert result["Train RMSE"] != pytest.approx(result["Train log-RMSE"])


def test_compare_models_sorts_by_validation_log_rmse_ascending():
    results = [
        {
            "Model": "Worst",
            "Train log-RMSE": 0.05,
            "Validation log-RMSE": 0.20,
            "Train RMSE": 100,
            "Validation RMSE": 200,
            "Train/Val Gap": 0.15,
        },
        {
            "Model": "Best",
            "Train log-RMSE": 0.05,
            "Validation log-RMSE": 0.10,
            "Train RMSE": 100,
            "Validation RMSE": 150,
            "Train/Val Gap": 0.05,
        },
        {
            "Model": "Middle",
            "Train log-RMSE": 0.05,
            "Validation log-RMSE": 0.15,
            "Train RMSE": 100,
            "Validation RMSE": 175,
            "Train/Val Gap": 0.10,
        },
    ]

    table = compare_models(results)

    assert list(table["Model"]) == [
        "Best",
        "Middle",
        "Worst",
    ]


def test_compare_models_index_is_reset():
    results = [
        {
            "Model": "A",
            "Train log-RMSE": 0.1,
            "Validation log-RMSE": 0.2,
            "Train RMSE": 1,
            "Validation RMSE": 2,
            "Train/Val Gap": 0.1,
        },
    ]

    table = compare_models(results)

    assert list(table.index) == [0]
