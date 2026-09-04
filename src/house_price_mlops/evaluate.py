"""
Regression evaluation for the Ames house-price project -- extracted from
notebooks/01_final_notebook_reviewed.ipynb, Stages 10.1 and 11, plus two additions
made when tuning the top 3 candidates (Lasso, Gradient Boosting, Hist
Gradient Boosting):

  - plot_model_diagnostics(): the 3-panel diagnostic plot (Actual vs
    Predicted, Residuals vs Predicted, Residual Distribution) that Stage
    10's notebook cells each built inline. Extracted here so it's written
    once, reused by every model, and testable.

  - cross_validate_models(): the mean/std K-fold cross-validation check
    used to confirm Lasso's single-split advantage held up, and again to
    compare the 3 tuned finalists. Extracted so the same logic used by
    hand in the notebook doesn't have to be retyped for the next model.

Regression-appropriate metrics only (log-RMSE, RMSE in dollars, the
train/validation gap as an overfitting signal) -- no classification
metrics anywhere. The primary metric is validation log-RMSE, matching
the competition's own evaluation metric.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def evaluate_model(model, model_name: str, X_train, y_train, X_val, y_val):
    """
    Fit `model` on train, evaluate on both train and validation.

    Parameters
    ----------
    model : an unfitted sklearn-style estimator (or Pipeline)
    model_name : str, used only for the returned result dict's label
    X_train, y_train, X_val, y_val : the split produced in notebook Stage 9.
        y_train/y_val are expected to already be log1p(SalePrice) --
        this function converts back to dollars internally for the
        human-readable RMSE, but always ranks/compares on the log scale.

    Returns
    -------
    result : dict of metrics (see keys below) -- append this to a list
        across models to build a comparison table with compare_models().
    fitted_model : the same model object, now fitted
    y_val_pred : predictions on X_val, log scale
    residuals : y_val - y_val_pred, log scale
    """
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_log_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_log_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    train_rmse = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(y_train_pred)))
    val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_val_pred)))

    residuals = y_val - y_val_pred

    result = {
        "Model": model_name,
        "Train log-RMSE": train_log_rmse,
        "Validation log-RMSE": val_log_rmse,
        "Train RMSE": train_rmse,
        "Validation RMSE": val_rmse,
        "Train/Val Gap": val_log_rmse - train_log_rmse,
    }
    return result, model, y_val_pred, residuals


def compare_models(all_results: list[dict]) -> pd.DataFrame:
    """
    Build the Stage 11 comparison table: every model's result, sorted by
    validation log-RMSE (ascending -- best first), matching the
    competition's own metric.
    """
    return (
        pd.DataFrame(all_results)
        .sort_values("Validation log-RMSE")
        .reset_index(drop=True)
    )


def plot_model_diagnostics(model_name: str, y_val, y_val_pred, residuals) -> plt.Figure:
    """
    The 3-panel diagnostic plot used for every model in notebook Stage 10:
      1. Actual vs Predicted, in dollars (expm1'd back from log scale)
      2. Residuals vs Predicted, log scale -- look for random scatter
         around zero; a funnel shape would indicate heteroscedasticity
      3. Residual distribution -- look for a roughly symmetric, bell
         shaped histogram centered on zero

    Returns the Figure so callers can further customize or save it;
    does NOT call plt.show() itself, so it works the same in a notebook
    (where the caller shows it) and in a test (where the caller closes it).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    actual_dollars = np.expm1(y_val)
    pred_dollars = np.expm1(y_val_pred)
    axes[0].scatter(actual_dollars, pred_dollars, alpha=0.5)
    lims = [
        min(actual_dollars.min(), pred_dollars.min()),
        max(actual_dollars.max(), pred_dollars.max()),
    ]
    axes[0].plot(lims, lims, linestyle="--", color="gray")
    axes[0].set_title(f"{model_name}: Actual vs Predicted ($)")
    axes[0].set_xlabel("Actual SalePrice ($)")
    axes[0].set_ylabel("Predicted SalePrice ($)")

    axes[1].scatter(y_val_pred, residuals, alpha=0.5)
    axes[1].axhline(0, linestyle="--", color="gray")
    axes[1].set_title(f"{model_name}: Residuals vs Predicted (log scale)")
    axes[1].set_xlabel("Predicted log(SalePrice)")
    axes[1].set_ylabel("Residual (log scale)")

    sns.histplot(residuals, kde=True, ax=axes[2])
    axes[2].set_title(f"{model_name}: Residual Distribution")
    axes[2].set_xlabel("Residual (log scale)")

    plt.tight_layout()
    return fig


def cross_validate_models(models: dict, X, y, cv) -> pd.DataFrame:
    """
    Run K-fold cross-validation for each model in `models`, reporting both
    the mean AND standard deviation of log-RMSE across folds -- the std is
    what a single train/validation split can never tell you: how much a
    model's score actually varies depending on which rows end up in which
    fold. A low mean with a high std is a less trustworthy result than a
    slightly higher mean with a low std.

    Parameters
    ----------
    models : dict of {name: unfitted sklearn-style estimator or Pipeline}
        e.g. the output of train.get_models(), or a dict built from
        train.get_tuning_configs() + train.build_tuned_model() results
    X, y : the FULL labeled dataset (not train/val split -- cross_val_score
        does its own internal splitting across `cv` folds)
    cv : a cross-validation splitter, e.g. KFold(n_splits=5, shuffle=True, random_state=42)

    Returns
    -------
    pd.DataFrame with columns [Model, CV log-RMSE (mean), CV log-RMSE (std)],
    sorted by mean ascending (best first).
    """
    rows = []
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="neg_root_mean_squared_error"
        )
        rows.append(
            {
                "Model": name,
                "CV log-RMSE (mean)": -scores.mean(),
                "CV log-RMSE (std)": scores.std(),
            }
        )
    return pd.DataFrame(rows).sort_values("CV log-RMSE (mean)").reset_index(drop=True)
