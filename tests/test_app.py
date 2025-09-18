import sys
import os
import pandas as pd

# Add project root to Python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import load_data  # now this should work


def test_load_data():
    """Test that load_data returns a non-empty pandas DataFrame."""
    df = load_data()
    assert isinstance(df, pd.DataFrame), "load_data should return a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"


def test_columns_exist():
    """Optional: Test that expected columns exist in the dataset."""
    df = load_data()
    expected_columns = ["bike_count", "date"]  # adjust to match your CSV
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' should exist in the dataset"


def test_load_data_with_filter():
    """Test that load_data returns a filtered DataFrame correctly."""
    df = load_data(start_date="2020-01-01", end_date="2021-01-31")
    assert not df.empty, "Filtered DataFrame should not be empty"
    assert df["date"].min() >= pd.to_datetime("2023-01-01")
    assert df["date"].max() <= pd.to_datetime("2023-01-31")
