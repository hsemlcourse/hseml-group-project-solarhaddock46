import os

import pandas as pd
import pytest

from preprocessing import (
    CAT_COLS,
    NUM_COLS,
    TARGET,
    clean_data,
    engineer_features,
    load_raw_data,
    split_data,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "adult.csv")
SEED = 42


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data(DATA_PATH)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


class TestLoadRawData:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_columns(self, raw_df):
        expected = set(CAT_COLS + NUM_COLS + [TARGET])
        assert expected.issubset(set(raw_df.columns))

    def test_min_rows(self, raw_df):
        assert len(raw_df) >= 30_000


class TestCleanData:
    def test_no_missing_values(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_no_duplicates(self, clean_df):
        assert clean_df.duplicated().sum() == 0

    def test_target_is_binary(self, clean_df):
        assert set(clean_df[TARGET].unique()).issubset({0, 1})

    def test_fewer_rows_than_raw(self, raw_df, clean_df):
        assert len(clean_df) < len(raw_df)

    def test_cat_cols_are_category(self, clean_df):
        for col in CAT_COLS:
            assert clean_df[col].dtype.name == "category", f"{col} not category"


class TestEngineerFeatures:
    def test_new_columns_added(self, clean_df):
        df = engineer_features(clean_df)
        assert "capital_net" in df.columns
        assert "age_group" in df.columns
        assert "is_usa" in df.columns

    def test_capital_net_formula(self, clean_df):
        df = engineer_features(clean_df)
        expected = clean_df["capital.gain"] - clean_df["capital.loss"]
        pd.testing.assert_series_equal(df["capital_net"], expected, check_names=False)

    def test_is_usa_is_binary(self, clean_df):
        df = engineer_features(clean_df)
        assert set(df["is_usa"].unique()).issubset({0, 1})

    def test_original_df_unchanged(self, clean_df):
        engineer_features(clean_df)
        assert "capital_net" not in clean_df.columns


class TestSplitData:
    def test_sizes_sum_to_total(self, clean_df):
        X = clean_df[CAT_COLS + NUM_COLS]
        y = clean_df[TARGET]
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=SEED)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)

    def test_no_index_overlap(self, clean_df):
        X = clean_df[CAT_COLS + NUM_COLS]
        y = clean_df[TARGET]
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=SEED)
        assert len(set(X_train.index) & set(X_val.index)) == 0
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(X_val.index) & set(X_test.index)) == 0

    def test_stratification(self, clean_df):
        X = clean_df[CAT_COLS + NUM_COLS]
        y = clean_df[TARGET]
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=SEED)
        global_rate = y.mean()
        for y_s in [y_train, y_val, y_test]:
            assert abs(y_s.mean() - global_rate) < 0.005

    def test_approximate_split_ratios(self, clean_df):
        X = clean_df[CAT_COLS + NUM_COLS]
        y = clean_df[TARGET]
        X_train, X_val, X_test, _, _, _ = split_data(X, y, seed=SEED)
        total = len(X)
        assert abs(len(X_train) / total - 0.70) < 0.01
        assert abs(len(X_val) / total - 0.15) < 0.01
        assert abs(len(X_test) / total - 0.15) < 0.01

    def test_reproducibility(self, clean_df):
        X = clean_df[CAT_COLS + NUM_COLS]
        y = clean_df[TARGET]
        splits_1 = split_data(X, y, seed=SEED)
        splits_2 = split_data(X, y, seed=SEED)
        for s1, s2 in zip(splits_1, splits_2):
            assert list(s1.index) == list(s2.index)
