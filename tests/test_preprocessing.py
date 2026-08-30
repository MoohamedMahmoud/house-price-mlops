"""
Tests for src/preprocessing.py (AmesCleaner).

Each test targets exactly ONE rule from AmesCleaner, using a small,
hand-built DataFrame instead of the real dataset — small enough that the
expected result is obvious just by reading the test, fast enough to run
on every commit.

Run with:
    uv run pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import pytest

from src.preprocessing import AmesCleaner


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

def _baseline_row(**overrides) -> dict:
    """
    One 'normal, nothing missing' house. Every test row below starts from
    this and overrides only the fields relevant to what it's testing —
    keeps every row realistic and keeps the test focused on one thing.
    """
    row = dict(
        Id=0,
        Neighborhood="CollgCr",
        LotFrontage=80.0,
        YearBuilt=2000,
        Alley="Pave",
        MasVnrType="BrkFace",
        MasVnrArea=150.0,
        BsmtQual="Gd",
        BsmtCond="TA",
        BsmtExposure="No",
        BsmtFinType1="GLQ",
        BsmtFinType2="Unf",
        FireplaceQu="Gd",
        GarageType="Attchd",
        GarageYrBlt=2000,
        GarageFinish="Fin",
        GarageQual="TA",
        GarageCond="TA",
        PoolQC="Ex",
        Fence="GdPrv",
        MiscFeature="Shed",
        Electrical="SBrkr",
    )
    row.update(overrides)
    return row


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    11 rows: 3 clean baseline rows (to seed medians/modes), plus one row
    per fill rule AmesCleaner needs to handle. Fit and transform are both
    run on this same DataFrame in most tests below — like fitting and
    cleaning a single training set.
    """
    rows = [
        # --- 3 baseline rows: give every "mode"/"median" something real to compute ---
        _baseline_row(Id=1, LotFrontage=80.0),
        _baseline_row(Id=2, LotFrontage=85.0),
        _baseline_row(Id=3, LotFrontage=75.0),

        # --- Row 4: MasVnrType exception — type missing, but area is real (200) ---
        _baseline_row(Id=4, MasVnrType=None, MasVnrArea=200.0),

        # --- Row 5: basement partial exception — only BsmtQual missing, rest present ---
        _baseline_row(Id=5, BsmtQual=None),

        # --- Row 6: garage partial exception — only GarageQual missing, rest present ---
        _baseline_row(Id=6, GarageQual=None),

        # --- Row 7: fully missing garage — structural "no garage", not an exception ---
        _baseline_row(
            Id=7, YearBuilt=1995,
            GarageType=None, GarageYrBlt=None, GarageFinish=None,
            GarageQual=None, GarageCond=None,
        ),

        # --- Row 8: plain structural fill — Alley missing ---
        _baseline_row(Id=8, Alley=None),

        # --- Row 9: Electrical missing (the one genuinely-unknown case) ---
        _baseline_row(Id=9, Electrical=None),

        # --- Row 10 + 11: LotFrontage missing in a DIFFERENT neighborhood,
        #     to prove the per-neighborhood median is actually used ---
        _baseline_row(Id=10, Neighborhood="Somerst", LotFrontage=None),
        _baseline_row(Id=11, Neighborhood="Somerst", LotFrontage=90.0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def cleaned(sample_df) -> pd.DataFrame:
    """fit() and transform() on the same data — the common case in these tests."""
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


def test_masvnr_type_exception_uses_mode_not_none(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 4].iloc[0]
    # area was real (200) -> must NOT be filled with 'None'
    assert row["MasVnrType"] == "BrkFace"  # the mode of all real values in this sample


def test_basement_partial_exception_uses_mode_not_none(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 5].iloc[0]
    # rest of the basement group was present -> basement is real, must NOT be 'None'
    assert row["BsmtQual"] == "Gd"


def test_garage_partial_exception_uses_mode_not_none(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 6].iloc[0]
    assert row["GarageQual"] == "TA"


def test_fully_missing_garage_gets_structural_none_and_yearbuilt(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 7].iloc[0]
    # ALL 5 garage columns were missing together -> this house has no garage
    assert row["GarageType"] == "None"
    assert row["GarageQual"] == "None"
    # the important one: filled with this row's OWN YearBuilt (1995), never 0
    assert row["GarageYrBlt"] == 1995


def test_alley_structural_fill(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 8].iloc[0]
    assert row["Alley"] == "None"


def test_electrical_filled_with_mode(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 9].iloc[0]
    assert row["Electrical"] == "SBrkr"


def test_lotfrontage_filled_with_neighborhood_median_not_global(cleaned, sample_df):
    row = cleaned.loc[sample_df["Id"] == 10].iloc[0]
    # Somerst's only other known frontage is 90 -> group median is 90,
    # NOT the global median (80) from the CollgCr-heavy rest of the sample.
    assert row["LotFrontage"] == 90.0


# ---------------------------------------------------------------------------
# fit/transform separation: transform() must reuse TRAIN statistics,
# never recompute anything from the data passed to transform()
# ---------------------------------------------------------------------------

def test_transform_reuses_train_statistics_not_test_data():
    train_df = pd.DataFrame([
        _baseline_row(Id=1, Neighborhood="CollgCr", LotFrontage=80.0),
        _baseline_row(Id=2, Neighborhood="CollgCr", LotFrontage=80.0),
    ])
    cleaner = AmesCleaner()
    cleaner.fit(train_df)

    # A neighborhood transform() has never seen before -> must fall back to
    # the TRAIN-wide global median (80), not silently do nothing or crash.
    test_df = pd.DataFrame([
        _baseline_row(Id=99, Neighborhood="StoneBr", LotFrontage=None),
    ])
    result = cleaner.transform(test_df)

    assert result.loc[0, "LotFrontage"] == 80.0


def test_transform_before_fit_raises_clear_error():
    cleaner = AmesCleaner()
    test_df = pd.DataFrame([_baseline_row(Id=1)])

    with pytest.raises(RuntimeError):
        cleaner.transform(test_df)