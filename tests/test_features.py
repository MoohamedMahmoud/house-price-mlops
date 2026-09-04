"""
Tests for src/features.py (AmesFeatureEngineer + apply_log_transform).

Uses the same small, hand-built DataFrame pattern as test_preprocessing.py.

Run with:
    uv run pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from house_price_mlops.features import (
    LOG_TRANSFORM_COLS,
    AmesFeatureEngineer,
    apply_log_transform,
)

# ---------------------------------------------------------------------------
# Shared sample data -- already-cleaned rows
# ---------------------------------------------------------------------------


def _clean_row(**overrides) -> dict:
    row = {
        "Neighborhood": "CollgCr",
        "GrLivArea": 1500.0,
        "LotArea": 8000.0,
        "MasVnrArea": 100.0,
        "OverallQual": 6,
        "ExterQual": "TA",
        "BsmtQual": "Gd",
        "BsmtExposure": "No",
        "GarageFinish": "Fin",
        "CentralAir": "Y",
        "LotShape": "Reg",
        "Functional": "Typ",
        "Alley": "None",
        "SalePrice": 180000.0,
    }

    row.update(overrides)

    return row


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _clean_row(Neighborhood="CollgCr"),
            _clean_row(Neighborhood="Somerst"),
            _clean_row(
                Neighborhood="OldTown",
                ExterQual="Gd",
                BsmtQual="TA",
                CentralAir="N",
            ),
        ]
    )


@pytest.fixture
def fitted_engineer(sample_df):
    fe = AmesFeatureEngineer()
    fe.fit(sample_df)
    return fe


# ---------------------------------------------------------------------------
# apply_log_transform
# ---------------------------------------------------------------------------


def test_log_transform_matches_manual_log1p():
    df = pd.DataFrame(
        {
            "GrLivArea": [1000.0, 2000.0],
            "OverallQual": [5, 8],
        }
    )

    result = apply_log_transform(
        df,
        columns=["GrLivArea"],
    )

    assert np.allclose(
        result["GrLivArea"],
        np.log1p([1000.0, 2000.0]),
    )

    assert list(result["OverallQual"]) == [5, 8]


def test_log_transform_handles_zero_without_error():
    df = pd.DataFrame(
        {
            "WoodDeckSF": [0.0, 100.0],
        }
    )

    result = apply_log_transform(
        df,
        columns=["WoodDeckSF"],
    )

    assert result.loc[0, "WoodDeckSF"] == 0.0


def test_default_columns_match_notebook_decision():
    expected = {
        "GrLivArea",
        "LotArea",
        "MasVnrArea",
        "1stFlrSF",
        "2ndFlrSF",
        "LotFrontage",
        "BsmtFinSF1",
        "WoodDeckSF",
        "OpenPorchSF",
    }

    assert set(LOG_TRANSFORM_COLS) == expected

    for rejected in [
        "TotalBsmtSF",
        "GarageArea",
        "BsmtUnfSF",
    ]:
        assert rejected not in LOG_TRANSFORM_COLS


# ---------------------------------------------------------------------------
# AmesFeatureEngineer -- ordinal / binary / one-hot
# ---------------------------------------------------------------------------


def test_ordinal_columns_encoded_as_integers(
    fitted_engineer,
    sample_df,
):
    result = fitted_engineer.transform(sample_df)

    assert result.loc[0, "ExterQual"] == 2
    assert result.loc[2, "ExterQual"] == 3


def test_binary_column_mapped_to_zero_one(
    fitted_engineer,
    sample_df,
):
    result = fitted_engineer.transform(sample_df)

    assert result.loc[0, "CentralAir"] == 1
    assert result.loc[2, "CentralAir"] == 0


def test_nominal_column_one_hot_encoded(
    fitted_engineer,
    sample_df,
):
    result = fitted_engineer.transform(sample_df)

    assert "Neighborhood_CollgCr" in result.columns
    assert "Neighborhood_Somerst" in result.columns
    assert "Neighborhood" not in result.columns

    assert result.loc[0, "Neighborhood_CollgCr"] == 1
    assert result.loc[0, "Neighborhood_Somerst"] == 0


def test_no_missing_values_after_transform(
    fitted_engineer,
    sample_df,
):
    result = fitted_engineer.transform(sample_df)

    assert result.isna().sum().sum() == 0


def test_row_order_and_index_preserved(
    fitted_engineer,
    sample_df,
):
    result = fitted_engineer.transform(sample_df)

    assert list(result.index) == list(sample_df.index)
    assert len(result) == len(sample_df)


def test_transform_before_fit_raises_clear_error():
    fe = AmesFeatureEngineer()

    with pytest.raises(RuntimeError):
        fe.transform(pd.DataFrame([_clean_row()]))


# ---------------------------------------------------------------------------
# fit/transform separation
# ---------------------------------------------------------------------------


def test_train_and_test_produce_identical_columns():
    train_df = pd.DataFrame(
        [
            _clean_row(Neighborhood="CollgCr"),
            _clean_row(Neighborhood="Somerst"),
            _clean_row(Neighborhood="OldTown"),
        ]
    )

    test_df = pd.DataFrame(
        [
            _clean_row(Neighborhood="CollgCr"),
        ]
    )

    fe = AmesFeatureEngineer()
    fe.fit(train_df)

    train_encoded = fe.transform(train_df)
    test_encoded = fe.transform(test_df)

    assert set(train_encoded.columns) == set(test_encoded.columns)

    assert test_encoded.loc[0, "Neighborhood_Somerst"] == 0
    assert test_encoded.loc[0, "Neighborhood_OldTown"] == 0
    assert test_encoded.loc[0, "Neighborhood_CollgCr"] == 1


def test_non_contiguous_index_does_not_introduce_nan():
    df = pd.DataFrame(
        [
            _clean_row(Neighborhood="CollgCr"),
            _clean_row(Neighborhood="Somerst"),
        ],
        index=[7, 42],
    )

    fe = AmesFeatureEngineer()
    fe.fit(df)

    result = fe.transform(df)

    assert result.isna().sum().sum() == 0
    assert list(result.index) == [7, 42]
