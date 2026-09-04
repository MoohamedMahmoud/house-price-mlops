"""
Cleaning pipeline for the Ames house-price dataset — Stage 2 (preprocessing) only.

Every decision here traces back to notebooks/01_eda_reviewed.ipynb, Section 3
(missing-value handling) and the two verified exceptions (MasVnrType/Area,
the basement group), generalized here via `audit_structural_missingness`.

This file deliberately does NOT include encoding (ordinal/one-hot) or any
column drop beyond `Id` — those are Stage 3 (feature engineering) decisions
that have not been made yet. See project_architecture.md for the full
pipeline map. Do not add encoding logic here until those decisions are
made and confirmed.

Usage:
    cleaner = AmesCleaner()
    cleaner.fit(train_df)                 # learns medians/modes from TRAIN ONLY
    train_clean = cleaner.transform(train_df)
    test_clean = cleaner.transform(test_df)  # same fitted params, never refit on test

The fit/transform split exists specifically to avoid training/serving skew:
any statistic used to fill a value must come from `fit()`, computed once on
train, and reused by `transform()` on anything that comes later.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Column groups, taken directly from data_description.txt / column_reference.md
# ---------------------------------------------------------------------------

# Columns where NA is a documented category ("no such feature"), not real missingness.
STRUCTURAL_NONE_COLS = [
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

# Column clusters describing one physical feature -- used to detect rows where
# the feature actually exists but only ONE field in the cluster is unrecorded
# (verified: 2 such rows in the basement group; 0 in the garage group).
FEATURE_GROUPS = [
    ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
    ["GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"],
]

# (categorical, numeric) pairs describing one feature -- used to detect rows
# where the category is unrecorded but the numeric side proves the feature
# exists (verified: 5 such rows for MasVnrType/MasVnrArea; 0 for the others).
PAIRED_CHECKS = [
    ("MasVnrType", "MasVnrArea"),
    ("GarageType", "GarageArea"),
    ("PoolQC", "PoolArea"),
    ("FireplaceQu", "Fireplaces"),
]

# Id is not a feature -- uncontroversial, dropped from the start.
# NOTE: Utilities is NOT dropped here. It was investigated (99.9% one value)
# but that is a Stage 3 decision that has not been confirmed yet.
DROP_COLS = ["Id"]


def audit_structural_missingness(
    df: pd.DataFrame,
    groups: list[list[str]] = FEATURE_GROUPS,
    paired: list[tuple[str, str]] = PAIRED_CHECKS,
) -> dict[str, pd.DataFrame]:
    """
    Systematic check for whether "NaN = feature absent" actually holds, instead
    of assuming it column-by-column. Run this on any new dataset before trusting
    a blanket structural-None fill.

    For each column GROUP describing one physical feature (e.g. all 5 Bsmt*
    columns): a row missing SOME but not ALL columns in the group means the
    feature exists and one attribute just wasn't recorded -- not "absent".

    For each (categorical, numeric) PAIR describing one feature: a NaN category
    next to a nonzero numeric value means the same thing.

    Returns {description: DataFrame of exception rows}. An empty dict means
    every candidate group/pair is internally consistent -- safe to blanket-fill.
    """
    exceptions: dict[str, pd.DataFrame] = {}

    for group in groups:
        present = [c for c in group if c in df.columns]
        if not present:
            continue
        na_count = df[present].isna().sum(axis=1)
        partial = df.loc[(na_count > 0) & (na_count < len(present)), present]
        if len(partial):
            exceptions[f"group inconsistency: {present}"] = partial

    for cat_col, num_col in paired:
        if cat_col not in df.columns or num_col not in df.columns:
            continue
        mismatch = df.loc[df[cat_col].isna() & df[num_col].fillna(0).ne(0)]
        if len(mismatch):
            exceptions[f"paired inconsistency: {cat_col} / {num_col}"] = mismatch[
                [cat_col, num_col]
            ]

    return exceptions


class AmesCleaner:
    """
    Stage 2 (preprocessing) only: missing-value handling and the Id drop.
    Fit on train, transform train/test identically. See module docstring.

    Deliberately has no encoding methods. Add those to a separate Stage 3
    class/module once the encoding scheme is actually decided.
    """

    def __init__(self) -> None:
        self.lotfrontage_by_neighborhood: pd.Series | None = None
        self.lotfrontage_global_median: float | None = None
        self.electrical_mode: str | None = None
        self.masvnr_type_mode: str | None = None
        self.feature_group_modes: dict[str, str] = {}
        self._fitted = False

    # ------------------------------------------------------------------ fit

    def fit(self, df: pd.DataFrame) -> AmesCleaner:
        """Learn every fill statistic from TRAIN data only."""
        df = df.copy()

        self.lotfrontage_by_neighborhood = df.groupby("Neighborhood")[
            "LotFrontage"
        ].median()
        self.lotfrontage_global_median = df["LotFrontage"].median()
        self.electrical_mode = df["Electrical"].mode()[0]
        self.masvnr_type_mode = df.loc[
            df["MasVnrType"].notna() & (df["MasVnrType"] != "None"), "MasVnrType"
        ].mode()[0]

        for group in FEATURE_GROUPS:
            for col in group:
                if col not in df.columns or col == "GarageYrBlt":
                    continue
                non_none = df.loc[df[col].notna() & (df[col] != "None"), col]
                if len(non_none):
                    self.feature_group_modes[col] = non_none.mode()[0]

        self._fitted = True
        return self

    # ------------------------------------------------------------ transform

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call .fit(train_df) before .transform()")
        return self._clean(df, is_fit=False)

    # ------------------------------------------------------------- helpers

    def _clean(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        """Missing-value handling + Id drop. Shared by fit() and transform()."""
        df = df.copy()

        # --- Step 0: fix the two verified exceptions BEFORE any blanket fill ---
        masvnr_mode = (
            self.masvnr_type_mode
            if not is_fit
            else df.loc[
                df["MasVnrType"].notna() & (df["MasVnrType"] != "None"), "MasVnrType"
            ].mode()[0]
        )
        masvnr_mismatch = df.index[
            df["MasVnrType"].isna() & df["MasVnrArea"].fillna(0).ne(0)
        ]
        df.loc[masvnr_mismatch, "MasVnrType"] = masvnr_mode

        for group in FEATURE_GROUPS:
            present = [c for c in group if c in df.columns]
            na_count = df[present].isna().sum(axis=1)
            partial_mask = (na_count > 0) & (na_count < len(present))
            for col in present:
                if col == "GarageYrBlt":
                    continue
                fix_idx = df.index[partial_mask & df[col].isna()]
                if len(fix_idx):
                    mode_val = self.feature_group_modes.get(col)
                    if mode_val is None and is_fit:
                        non_none = df.loc[df[col].notna() & (df[col] != "None"), col]
                        mode_val = non_none.mode()[0] if len(non_none) else "None"
                    df.loc[fix_idx, col] = mode_val

        # --- Step 1: structural "feature absent" fills ---
        df[STRUCTURAL_NONE_COLS] = df[STRUCTURAL_NONE_COLS].fillna("None")

        electrical_fill = (
            self.electrical_mode if not is_fit else df["Electrical"].mode()[0]
        )
        df["Electrical"] = df["Electrical"].fillna(electrical_fill)

        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        # GarageYrBlt: NaN means no garage; 0 is not a valid year, so use the
        # house's own build year instead.
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

        # LotFrontage: per-neighborhood median, fitted on train; fall back to
        # the train-wide median for any neighborhood transform() sees that
        # fit() never saw.
        if is_fit:
            by_nb = df.groupby("Neighborhood")["LotFrontage"].transform("median")
            global_med = df["LotFrontage"].median()
        else:
            by_nb = df["Neighborhood"].map(self.lotfrontage_by_neighborhood)
            global_med = self.lotfrontage_global_median
        df["LotFrontage"] = df["LotFrontage"].fillna(by_nb).fillna(global_med)

        # --- Step 2: drop Id only (Utilities intentionally NOT dropped here) ---
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        return df
