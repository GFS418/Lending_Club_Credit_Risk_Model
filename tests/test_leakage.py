"""The core discipline of the project, enforced programmatically.

If any post-origination (leakage) column ever reaches the model feature list,
CI fails. These tests read the committed reports/data_dictionary.csv, so they
run without the (gitignored) database.
"""
import data_prep as dp
import lc_data_dictionary as lcd


def test_known_leakage_columns_are_flagged():
    for col in ["recoveries", "total_pymnt", "out_prncp", "total_rec_prncp",
                "last_fico_range_high", "last_pymnt_amnt", "next_pymnt_d"]:
        assert lcd.annotate(col)[0] == "LEAKAGE", col


def test_prefix_rules_flag_leakage_families():
    for col in ["hardship_type", "settlement_amount", "deferral_term"]:
        assert lcd.annotate(col)[0] == "LEAKAGE", col


def test_target_and_lc_model_flags():
    assert lcd.annotate("loan_status")[0] == "TARGET"
    for col in ["grade", "sub_grade", "int_rate"]:
        assert lcd.annotate(col)[0] == "LC_MODEL", col


def test_unknown_column_defaults_to_review():
    assert lcd.annotate("some_unknown_col")[0] == "REVIEW"


def test_no_leakage_feature_reaches_the_model():
    """The headline guard: no LEAKAGE-flagged column may be a model feature."""
    leaked = [c for c in dp.feature_columns() if lcd.annotate(c)[0] == "LEAKAGE"]
    assert leaked == [], f"leakage columns leaked into features: {leaked}"


def test_target_split_and_zero_variance_excluded_from_features():
    feats = set(dp.feature_columns())
    for col in ["loan_status", "issue_d", "earliest_cr_line", "policy_code"]:
        assert col not in feats, col
