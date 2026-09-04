"""
Tests for src/preprocessing.py (AmesCleaner).

Each test targets exactly ONE rule from AmesCleaner, using a small,
hand-built DataFrame instead of the real dataset -- small enough that the
expected result is obvious just by reading the test, fast enough to run
on every commit.

Run with:
    uv run pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import pytest

from house_price_mlops.preprocessing import AmesCleaner

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------


def _baseline_row(**overrides) -> dict:
    """
    One 'normal, nothing missing' house. Every test row below starts from
    this and overrides only the fields relevant to what it's testing.
    """
    row = {
        "Id": 0,
        "Neighborhood": "CollgCr",
        "LotFrontage": 80.0,
        "YearBuilt": 2000,
        "Alley": "Pave",
        "MasVnrType": "BrkFace",
        "MasVnrArea": 150.0,
        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinType2": "Unf",
        "FireplaceQu": "Gd",
        "GarageType": "Attchd",
        "GarageYrBlt": 2000,
        "GarageFinish": "Fin",
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PoolQC": "Ex",
        "Fence": "GdPrv",
        "MiscFeature": "Shed",
        "Electrical": "SBrkr",
    }

    row.update(overrides)

    return row


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    11 rows: 3 clean baseline rows to seed medians/modes, plus rows
    covering the fill rules AmesCleaner needs to handle.
    """
    rows = [
        _baseline_row(
            Id=1,
            LotFrontage=80.0,
        ),
        _baseline_row(
            Id=2,
            LotFrontage=85.0,
        ),
        _baseline_row(
            Id=3,
            LotFrontage=75.0,
        ),
        _baseline_row(
            Id=4,
            MasVnrType=None,
            MasVnrArea=200.0,
        ),
        _baseline_row(
            Id=5,
            BsmtQual=None,
        ),
        _baseline_row(
            Id=6,
            GarageQual=None,
        ),
        _baseline_row(
            Id=7,
            YearBuilt=1995,
            GarageType=None,
            GarageYrBlt=None,
            GarageFinish=None,
            GarageQual=None,
            GarageCond=None,
        ),
        _baseline_row(
            Id=8,
            Alley=None,
        ),
        _baseline_row(
            Id=9,
            Electrical=None,
        ),
        _baseline_row(
            Id=10,
            Neighborhood="Somerst",
            LotFrontage=None,
        ),
        _baseline_row(
            Id=11,
            Neighborhood="Somerst",
            LotFrontage=90.0,
        ),
    ]

    return pd.DataFrame(rows)


@pytest.fixture
def cleaned(sample_df) -> pd.DataFrame:
    """Fit and transform on the same DataFrame."""
    cleaner = AmesCleaner()
    cleaner.fit(sample_df)

    return cleaner.transform(sample_df)


# ---------------------------------------------------------------------------
# One test per rule
# ---------------------------------------------------------------------------


def test_no_missing_values_after_clean(cleaned):
    assert cleaned.isna().sum().sum() == 0


def test_id_column_is_dropped(cleaned):
    assert "Id" not in cleaned.columns


def test_masvnr_type_exception_uses_mode_not_none(
    cleaned,
    sample_df,
):
    row = cleaned.loc[sample_df["Id"] == 4].iloc[0]

    assert row["MasVnrType"] == "BrkFace"


def test_basement_partial_exception_uses_mode_not_none(
    cleaned,
    sample_df,
):
    row = cleaned.loc[sample_df["Id"] == 5].iloc[0]

    assert row["BsmtQual"] == "Gd"


def test_garage_partial_exception_uses_mode_not_none(
    cleaned,
    sample_df,
):
    row = cleaned.loc[sample_df["Id"] == 6].iloc[0]

    assert row["GarageQual"] == "TA"


def test_fully_missing_garage_gets_structural_none_and_yearbuilt(
    cleaned,
    sample_df,
):
    row = cleaned.loc[sample_df["Id"] == 7].iloc[0]

    assert row["GarageType"] == "None"
    assert row["GarageQual"] == "None"
    assert row["GarageYrBlt"] == 1995


def test_alley_structural_fill(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 8].iloc[0]

    assert row["Alley"] == "None"


def test_electrical_filled_with_mode(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 9].iloc[0]

    assert row["Electrical"] == "SBrkr"


def test_lotfrontage_filled_with_neighborhood_median_not_global(
    cleaned,
    sample_df,
):
    row = cleaned.loc[sample_df["Id"] == 10].iloc[0]

    assert row["LotFrontage"] == 90.0


# ---------------------------------------------------------------------------
# fit/transform separation
# ---------------------------------------------------------------------------


def test_transform_reuses_train_statistics_not_test_data():
    train_df = pd.DataFrame(
        [
            _baseline_row(
                Id=1,
                Neighborhood="CollgCr",
                LotFrontage=80.0,
            ),
            _baseline_row(
                Id=2,
                Neighborhood="CollgCr",
                LotFrontage=80.0,
            ),
        ]
    )

    cleaner = AmesCleaner()
    cleaner.fit(train_df)

    test_df = pd.DataFrame(
        [
            _baseline_row(
                Id=99,
                Neighborhood="StoneBr",
                LotFrontage=None,
            ),
        ]
    )

    result = cleaner.transform(test_df)

    assert result.loc[0, "LotFrontage"] == 80.0


def test_transform_before_fit_raises_clear_error():
    cleaner = AmesCleaner()

    test_df = pd.DataFrame(
        [
            _baseline_row(Id=1),
        ]
    )

    with pytest.raises(RuntimeError):
        cleaner.transform(test_df)
