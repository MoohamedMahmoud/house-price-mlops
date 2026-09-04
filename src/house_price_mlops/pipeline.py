"""
End-to-end dataset assembly for the Ames house-price project.

This module exists specifically to fix a real bug found during notebook
review: the Stage 7 outlier keep/drop decision was computed and printed,
but never actually applied when building the final modeling dataframe --
`final_df` was always built from every row regardless of what the
experiment concluded. It was invisible in that run only because the
decision happened to be KEEP (a no-op). Had it concluded DROP, the model
would have silently kept the outliers anyway.

build_dataset() fixes this by making the decision an explicit, required
parameter -- there is no way to call it without stating keep_outliers.

Usage:
    from src.pipeline import build_dataset

    X_train, y_train, cleaner, engineer = build_dataset(
        raw_train_df, keep_outliers=True, fit=True
    )
    X_val, y_val, _, _ = build_dataset(
        raw_val_df, keep_outliers=True, fit=False,
        cleaner=cleaner, engineer=engineer,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from house_price_mlops.features import AmesFeatureEngineer
from house_price_mlops.preprocessing import AmesCleaner


def find_outlier_ids(raw_df: pd.DataFrame) -> list:
    """
    The two documented GrLivArea/SalePrice anomalies (notebook Stage 7):
    houses with GrLivArea > 4000 AND SalePrice < 300000. Deliberately
    combines both conditions -- GrLivArea alone would also catch two
    large, legitimately expensive houses (4316/4476 sqft, $755k/$745k)
    that fit the trend fine and are NOT outliers.

    Must be called on RAW (or at least un-log-transformed) data -- if
    GrLivArea has already been through log1p, this threshold is meaningless.
    """
    mask = (raw_df["GrLivArea"] > 4000) & (raw_df["SalePrice"] < 300000)
    return (
        raw_df.loc[mask, "Id"].tolist()
        if "Id" in raw_df.columns
        else raw_df.index[mask].tolist()
    )


def build_dataset(
    raw_df: pd.DataFrame,
    keep_outliers: bool,
    fit: bool,
    cleaner: AmesCleaner | None = None,
    engineer: AmesFeatureEngineer | None = None,
):
    """
    Full pipeline: raw data -> cleaned -> (outlier decision applied) ->
    encoded -> (X, y) ready for a model.

    Parameters
    ----------
    raw_df : raw dataframe (must still have an 'Id' column and 'SalePrice')
    keep_outliers : bool, REQUIRED, no default -- this is deliberate. The
        exact bug this module exists to prevent was a decision that had a
        silent default (effectively "always keep") because nothing forced
        it to be stated. Forcing the caller to pass this explicitly means
        it can never again be silently ignored.
    fit : bool -- True for training data (fits new AmesCleaner/
        AmesFeatureEngineer instances and returns them). False for
        validation/test data (must pass in the cleaner/engineer already
        fitted on training data -- never refit on non-training data).
    cleaner, engineer : required when fit=False; ignored (and freshly
        created) when fit=True.

    Returns
    -------
    X : pd.DataFrame, features only, SalePrice column removed
    y : pd.Series, log1p(SalePrice)
    cleaner : the fitted AmesCleaner (same one passed in, if fit=False)
    engineer : the fitted AmesFeatureEngineer (same one passed in, if fit=False)
    """
    if not fit and (cleaner is None or engineer is None):
        raise ValueError(
            "fit=False requires passing in an already-fitted cleaner and "
            "engineer (from a prior fit=True call on training data). "
            "Fitting fresh statistics on non-training data would be "
            "training/serving skew."
        )

    outlier_ids = find_outlier_ids(raw_df) if not keep_outliers else []

    if fit:
        cleaner = AmesCleaner()
        cleaner.fit(raw_df)
    clean_df = cleaner.transform(raw_df)

    if not keep_outliers and outlier_ids:
        id_col = "Id" if "Id" in raw_df.columns else None
        if id_col:
            drop_mask = raw_df["Id"].isin(outlier_ids)
        else:
            drop_mask = raw_df.index.isin(outlier_ids)
        clean_df = clean_df.loc[~drop_mask.values]

    if fit:
        engineer = AmesFeatureEngineer()
        engineer.fit(clean_df)
    encoded = engineer.transform(clean_df)

    y = np.log1p(encoded["SalePrice"])
    drop_cols = [c for c in ("SalePrice", "Id") if c in encoded.columns]
    X = encoded.drop(columns=drop_cols)

    return X, y, cleaner, engineer
