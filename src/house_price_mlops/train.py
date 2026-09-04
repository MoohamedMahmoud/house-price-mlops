"""
Model definitions and final training entry point for the Ames house-price
project.

The ten baseline models and tuning configurations are kept consistent with
the reviewed notebook.

The final model selected for the current project is:
    Lasso
    alpha = 0.003

The final artifact stores:
    - trained model
    - fitted AmesCleaner
    - fitted AmesFeatureEngineer
    - exact training feature columns
    - required raw input columns
    - model metadata
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from house_price_mlops.pipeline import build_dataset

# ============================================================
# Project paths / final model configuration
# ============================================================

RAW_TRAIN_PATH = Path("data/raw/train.csv")
MODEL_PATH = Path("models/lasso_model.joblib")

FINAL_MODEL_NAME = "Lasso"
FINAL_MODEL_PARAMS = {
    "model__alpha": 0.003,
}

# The notebook decision was to KEEP the two genuine GrLivArea/SalePrice
# anomalies rather than remove them.
KEEP_OUTLIERS = True


# ============================================================
# Model definitions
# ============================================================


def get_models() -> dict:
    """Return the ten baseline models used in the notebook."""
    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.0005, max_iter=10000)),
            ]
        ),
        "Elastic Net": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.0005,
                        l1_ratio=0.5,
                        max_iter=10000,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42,
        ),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]
        ),
    }


# ============================================================
# Hyperparameter tuning
# ============================================================


def get_tuning_configs() -> dict:
    """Return tuning estimators and grids for the three selected candidates."""
    return {
        "Lasso": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Lasso(max_iter=10000)),
                ]
            ),
            "param_grid": {
                "model__alpha": [
                    0.0001,
                    0.0003,
                    0.0005,
                    0.001,
                    0.003,
                    0.005,
                    0.01,
                    0.03,
                    0.05,
                    0.1,
                ]
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 4],
            },
        },
        "Hist Gradient Boosting": {
            "estimator": HistGradientBoostingRegressor(
                random_state=42,
            ),
            "param_grid": {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_iter": [100, 200, 300],
                "max_depth": [None, 3, 5],
            },
        },
    }


def tune_models(X_train, y_train, cv, n_jobs: int = -1) -> dict:
    """Fit the three GridSearchCV searches on training data only."""
    searches = {}

    for name, config in get_tuning_configs().items():
        search = GridSearchCV(
            estimator=clone(config["estimator"]),
            param_grid=config["param_grid"],
            cv=cv,
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=n_jobs,
        )

        search.fit(X_train, y_train)
        searches[name] = search

    return searches


# ============================================================
# Model construction / training
# ============================================================


def build_tuned_model(model_name: str, best_params: dict):
    """Build an estimator from parameters returned by GridSearchCV.

    This avoids hard-coding an entire GridSearchCV object into the final
    artifact. The selected parameters are explicitly provided here.
    """
    configs = get_tuning_configs()

    if model_name not in configs:
        raise KeyError(f"Unknown tuned model: {model_name}")

    model = clone(configs[model_name]["estimator"])
    model.set_params(**best_params)

    return model


def train_model(model, X_train, y_train):
    """Fit one estimator and return it."""
    model.fit(X_train, y_train)
    return model


# ============================================================
# Final model training + artifact creation  Cause of Docker and files arrangements in the src/house_price_mlops
# ============================================================


def train_final_model() -> dict:
    """Train the final Lasso model and save the complete artifact.

    The artifact contains everything required for serving:
        model
        cleaner
        engineer
        feature_columns
        required_input_columns
        model_name
        model_params
        keep_outliers
    """
    if not RAW_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {RAW_TRAIN_PATH}")

    print(f"Loading training data from: {RAW_TRAIN_PATH}")

    raw_train = pd.read_csv(RAW_TRAIN_PATH)

    print(f"Training rows: {len(raw_train)}")
    print(f"Raw columns: {len(raw_train.columns)}")

    # --------------------------------------------------------
    # Build the complete training dataset.
    #
    # build_dataset:
    #   - fits AmesCleaner
    #   - fits AmesFeatureEngineer
    #   - handles the documented outlier decision
    #   - returns model-ready X/y plus fitted preprocessing objects
    # --------------------------------------------------------

    X_train, y_train, cleaner, engineer = build_dataset(
        raw_train,
        keep_outliers=KEEP_OUTLIERS,
        fit=True,
    )

    print(f"Processed training rows: {len(X_train)}")
    print(f"Processed features: {X_train.shape[1]}")

    # --------------------------------------------------------
    # Build and train the selected final Lasso model
    # --------------------------------------------------------

    model = build_tuned_model(
        FINAL_MODEL_NAME,
        FINAL_MODEL_PARAMS,
    )

    model = train_model(
        model,
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Record the raw input columns expected by the API.
    #
    # Id and SalePrice are not prediction inputs.
    # --------------------------------------------------------

    required_input_columns = [
        column for column in raw_train.columns if column not in {"Id", "SalePrice"}
    ]

    # --------------------------------------------------------
    # Store everything needed for inference.
    # --------------------------------------------------------

    artifact = {
        "model": model,
        "cleaner": cleaner,
        "engineer": engineer,
        "feature_columns": list(X_train.columns),
        "required_input_columns": required_input_columns,
        "model_name": FINAL_MODEL_NAME,
        "model_params": FINAL_MODEL_PARAMS,
        "keep_outliers": KEEP_OUTLIERS,
    }

    # Make sure models/ exists.
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save artifact.
    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print()
    print("=" * 60)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {FINAL_MODEL_NAME}")
    print(f"Parameters: {FINAL_MODEL_PARAMS}")
    print(f"Training rows: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Keep outliers: {KEEP_OUTLIERS}")
    print(f"Artifact: {MODEL_PATH}")
    print("=" * 60)

    return artifact


# ============================================================
# Module entry point
# ============================================================


def main() -> None:
    """Entry point for ``python -m house_price_mlops.train``."""
    train_final_model()


if __name__ == "__main__":
    main()
