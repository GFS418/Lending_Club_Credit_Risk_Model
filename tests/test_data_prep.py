"""Unit tests for the label, type-parsing, filtering, and split logic.

All run on small synthetic frames — no database required.
"""
import numpy as np
import pandas as pd

import data_prep as dp


def test_build_label_keeps_finished_and_maps_default():
    df = pd.DataFrame({"loan_status": [
        "Fully Paid", "Charged Off", "Default", "Current", "Late (31-120 days)"]})
    out = dp.build_label(df)
    assert len(out) == 3                       # in-progress dropped
    assert out["default"].tolist() == [0, 1, 1]


def test_parse_types_term_emp_length_and_credit_age():
    df = pd.DataFrame({
        "term": [" 36 months", " 60 months"],
        "emp_length": ["10+ years", "< 1 year"],
        "issue_d": ["Dec-2015", "Jan-2016"],
        "earliest_cr_line": ["Dec-2010", "Jan-2016"],
    })
    out = dp.parse_types(df)
    assert out["term"].tolist() == [36.0, 60.0]
    assert out["emp_length"].tolist() == [10.0, 0.0]
    assert out["credit_age_months"].tolist() == [60.0, 0.0]
    assert "earliest_cr_line" not in out.columns   # raw date dropped
    assert "issue_dt" in out.columns               # retained for splitting


def test_parse_types_emp_length_na_maps_to_nan():
    df = pd.DataFrame({"term": [" 36 months"], "emp_length": ["n/a"],
                       "issue_d": ["Dec-2015"], "earliest_cr_line": ["Dec-2010"]})
    out = dp.parse_types(df)
    assert np.isnan(out["emp_length"].iloc[0])


def test_maturity_cutoff_drops_later_vintages():
    df = pd.DataFrame({"issue_dt": pd.to_datetime(
        ["2015-06-30", "2016-03-31", "2017-01-31"])})
    out = dp.apply_maturity_cutoff(df, cutoff="2016-12")
    assert len(out) == 2
    assert out["issue_dt"].max() <= pd.Timestamp("2016-12-31")


def test_time_split_is_out_of_time():
    df = pd.DataFrame({
        "issue_dt": pd.to_datetime(["2014-06-30", "2015-06-30", "2016-06-30"]),
        "x": [1, 2, 3]})
    train, test = dp.time_split(df, split_date="2015-12")
    assert train["x"].tolist() == [1, 2]       # issued <= 2015-12
    assert test["x"].tolist() == [3]           # issued after


def test_split_feature_types_by_numeric():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]})
    nums, cats = dp.split_feature_types(df, ["a", "b", "c"])
    assert set(nums) == {"a", "c"}
    assert cats == ["b"]
