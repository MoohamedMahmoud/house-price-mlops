"""
Feature preparation for the Ames house-price dataset -- Stage 5-6 logic,
extracted from notebooks/01_final_notebook_reviewed.ipynb.

Every value here (which columns get log1p, every ordinal mapping) traces
directly to a decided, evidenced result in that notebook -- see:
  - Stage 5.2: skew-driven log1p column selection (cell output)
  - Stage 6: ordinal/binary/nominal encoding decisions

One addition beyond what the notebook's plain `encode()` function did:
this module uses a fit/transform split (like AmesCleaner) so the one-hot
vocabulary is learned ONCE on train and reused on any other data --
guaranteeing train and test always produce identical columns in the same
order, even if test.csv contains categories in a different frequency or
is missing one your training data happened to have. The notebook's
version didn't need this (it only ever encoded one dataset at a time),
but src/ code does, to avoid training/serving skew.

Usage:
    fe = AmesFeatureEngineer()
    fe.fit(train_clean)                 # learns one-hot vocabulary from TRAIN ONLY
    train_encoded = fe.transform(train_clean)
    test_encoded = fe.transform(test_clean)   # same vocabulary, never refit
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Decided in notebook Stage 5.2 -- columns where log1p measurably reduced
# skew (|raw_skew| > 0.75 AND log1p skew at least 30% smaller). Excluded:
# TotalBsmtSF, GarageArea, BsmtUnfSF -- log1p made their skew WORSE
# (zero-inflated columns; see notebook for the actual before/after numbers).
# ---------------------------------------------------------------------------
LOG_TRANSFORM_COLS = [
    "GrLivArea",
    "LotArea",
    "MasVnrArea",
    "1stFlrSF",
    "2ndFlrSF",
    "LotFrontage",
    "BsmtFinSF1",
    "WoodDeckSF",
    "OpenPorchSF",
]

# Decided in notebook Stage 6 -- fixed scales taken from data_description.txt,
# not learned from data (the order is a documented fact, e.g. Po<Fa<TA<Gd<Ex).
ORDINAL_MAPS = {
    "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {
        "None": 0,
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6,
    },
    "BsmtFinType2": {
        "None": 0,
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6,
    },
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
    "Functional": {
        "Sal": 0,
        "Sev": 1,
        "Maj2": 2,
        "Maj1": 3,
        "Mod": 4,
        "Min2": 5,
        "Min1": 6,
        "Typ": 7,
    },
}

# Decided in notebook Stage 6 -- CentralAir is genuinely binary (Y/N).
BINARY_MAP = {"CentralAir": {"N": 0, "Y": 1}}


def apply_log_transform(
    df: pd.DataFrame, columns: list[str] = LOG_TRANSFORM_COLS
) -> pd.DataFrame:
    """log1p the decided columns. Safe for zeros (log1p(0) = 0); not valid
    for negative values -- none exist among these columns in this dataset."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


class AmesFeatureEngineer:
    """
    Stage 6 encoding: ordinal maps (fixed), binary map (fixed), one-hot
    for everything else (vocabulary learned from TRAIN only).
    Fit on train, transform train/test identically -- see module docstring.
    """

    def __init__(self) -> None:
        self.onehot_categories: dict[str, list[str]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> AmesFeatureEngineer:
        """Learn the one-hot vocabulary from TRAIN data only."""
        df = self._apply_fixed_encodings(df)
        skip = (
            set(ORDINAL_MAPS) | set(BINARY_MAP) | {"SalePrice", "SalePrice_log", "Id"}
        )
        nominal_cols = [
            c
            for c in df.select_dtypes(include=["object", "string"]).columns
            if c not in skip
        ]
        for col in nominal_cols:
            self.onehot_categories[col] = sorted(df[col].dropna().unique().tolist())
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call .fit(train_df) before .transform()")
        df = self._apply_fixed_encodings(df)
        for col, categories in self.onehot_categories.items():
            if col not in df.columns:
                continue
            # IMPORTANT: build dummies from df[col] itself (which carries df's
            # real index), not from a bare pd.Categorical array (which resets
            # to a fresh 0..N-1 index). Using the bare-array version silently
            # misaligns rows on concat whenever df's index isn't already a
            # plain 0..N-1 range -- which it never is after train_test_split.
            typed_col = df[col].astype(pd.CategoricalDtype(categories=categories))
            dummies = pd.get_dummies(typed_col, prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def _apply_fixed_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """log1p + ordinal + binary -- all fixed, need no fitting."""
        df = apply_log_transform(df)
        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        for col, mapping in BINARY_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        return df
