"""
Tests for src/pipeline.py.

The most important tests here are the ones directly regression-testing
the bug found in notebook review: keep_outliers must actually change the
output, not just be accepted and ignored.

Run with:
    uv run pytest tests/test_pipeline.py -v
"""

import inspect

import pandas as pd
import pytest

from house_price_mlops.pipeline import build_dataset, find_outlier_ids


def _make_row(
    id_,
    gr_liv_area,
    sale_price,
    **overrides,
):
    row = {
        "Id": id_,
        "GrLivArea": gr_liv_area,
        "SalePrice": sale_price,
        "Neighborhood": "CollgCr",
        "LotFrontage": 80.0,
        "YearBuilt": 2000,
        "Alley": "None",
        "MasVnrType": "None",
        "MasVnrArea": 0.0,
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinType2": "Unf",
        "BsmtFinSF1": 0.0,
        "FireplaceQu": "None",
        "GarageType": "Attchd",
        "GarageYrBlt": 2000.0,
        "GarageFinish": "Fin",
        "GarageQual": "TA",
        "GarageCond": "TA",
        "GarageArea": 400.0,
        "PoolQC": "None",
        "Fence": "None",
        "MiscFeature": "None",
        "Electrical": "SBrkr",
        "ExterQual": "TA",
        "ExterCond": "TA",
        "HeatingQC": "TA",
        "KitchenQual": "TA",
        "OverallQual": 6,
        "TotalBsmtSF": 800.0,
        "LotArea": 8000.0,
        "WoodDeckSF": 0.0,
        "OpenPorchSF": 0.0,
        "CentralAir": "Y",
        "Functional": "Typ",
        "PavedDrive": "Y",
        "LotShape": "Reg",
        "LandSlope": "Gtl",
        "BsmtUnfSF": 200.0,
        "HouseStyle": "1Story",
        "MSZoning": "RL",
        "_1stFlrSF": 800.0,
        "_2ndFlrSF": 0.0,
    }

    row.update(overrides)

    row["1stFlrSF"] = row.pop("_1stFlrSF")
    row["2ndFlrSF"] = row.pop("_2ndFlrSF")

    return row


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rows = [
        _make_row(
            1,
            1500,
            180000,
            MasVnrType="BrkFace",
            MasVnrArea=120.0,
        ),
        _make_row(2, 1600, 190000),
        _make_row(3, 1400, 170000),
        _make_row(4, 1700, 200000),
        _make_row(
            5,
            4676,
            184750,
            OverallQual=10,
        ),
        _make_row(
            6,
            5642,
            160000,
            OverallQual=10,
        ),
        _make_row(
            7,
            4316,
            755000,
            OverallQual=10,
        ),
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# find_outlier_ids
# ---------------------------------------------------------------------------


def test_find_outlier_ids_catches_only_genuine_anomalies(
    sample_df,
):
    ids = find_outlier_ids(sample_df)

    assert set(ids) == {5, 6}


def test_find_outlier_ids_does_not_catch_large_expensive_house(
    sample_df,
):
    ids = find_outlier_ids(sample_df)

    assert 7 not in ids


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_keep_outliers_true_keeps_all_rows(sample_df):
    X, _, _, _ = build_dataset(
        sample_df,
        keep_outliers=True,
        fit=True,
    )

    assert len(X) == len(sample_df)


def test_keep_outliers_false_actually_drops_the_rows(
    sample_df,
):
    X, y, _, _ = build_dataset(
        sample_df,
        keep_outliers=False,
        fit=True,
    )

    assert len(X) == len(sample_df) - 2
    assert len(y) == len(sample_df) - 2


def test_keep_outliers_is_a_required_argument():
    sig = inspect.signature(build_dataset)

    assert sig.parameters["keep_outliers"].default is inspect.Parameter.empty


def test_fit_false_without_cleaner_raises_clear_error(
    sample_df,
):
    with pytest.raises(ValueError):
        build_dataset(
            sample_df,
            keep_outliers=True,
            fit=False,
        )


def test_no_missing_values_in_output(sample_df):
    X, y, _, _ = build_dataset(
        sample_df,
        keep_outliers=True,
        fit=True,
    )

    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0


def test_no_target_leakage_in_X(sample_df):
    X, _, _, _ = build_dataset(
        sample_df,
        keep_outliers=True,
        fit=True,
    )

    assert not any("SalePrice" in column for column in X.columns)

    assert "Id" not in X.columns


def test_fit_false_reuses_fitted_cleaner_and_engineer(
    sample_df,
):
    """
    fit=False must reuse the cleaner and engineer fitted on training data
    rather than creating new statistics from validation data.
    """
    X_train, _, cleaner, engineer = build_dataset(
        sample_df,
        keep_outliers=True,
        fit=True,
    )

    val_df = sample_df.iloc[:2].copy()

    X_val, _, cleaner_out, engineer_out = build_dataset(
        val_df,
        keep_outliers=True,
        fit=False,
        cleaner=cleaner,
        engineer=engineer,
    )

    assert cleaner_out is cleaner
    assert engineer_out is engineer
    assert set(X_val.columns) == set(X_train.columns)
