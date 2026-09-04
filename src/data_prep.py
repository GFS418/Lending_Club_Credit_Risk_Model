"""
Session 2 data prep: label, maturity filter, out-of-time split, preprocessing.

The feature list is driven by the triage decisions in
reports/data_dictionary.csv (keep_decision == 'keep'), so the model stays
in sync with the analyst's keep/drop/leakage worksheet.

Key design decisions (see README):
  * Label: Charged Off / Default -> 1, Fully Paid -> 0; everything else dropped.
  * Maturity cutoff: drop loans issued after MATURITY_CUTOFF to limit the
    survivorship bias from unresolved recent vintages.
  * Out-of-time split: train on older vintages, test on newer (by issue_d).
  * grade/sub_grade/int_rate are held out of training (KFCDT: "keep for
    comparison, don't train on") for a later with/without-LC comparison.
  * All transformers are fit on TRAIN ONLY (sklearn Pipeline) to avoid
    leaking test statistics into fitting.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "lending_club.db"
DICT_PATH = ROOT / "reports" / "data_dictionary.csv"

TARGET_COL = "loan_status"
FINISHED_GOOD = {"Fully Paid"}
FINISHED_BAD = {"Charged Off", "Default"}
FINISHED = FINISHED_GOOD | FINISHED_BAD

# Held out of the model, kept for later comparison to LC's own risk grade.
KFCDT_COLS = ["grade", "sub_grade", "int_rate"]

# In the keep-list but NOT usable as direct model features:
#   issue_d          -> defines the time split; calendar time doesn't
#                       generalize out-of-time, so not a feature
#   earliest_cr_line -> raw date; replaced by derived credit_age_months
#   policy_code      -> constant (==1) in this data; zero variance
NON_FEATURE_KEEPS = {"issue_d", "earliest_cr_line", "policy_code", TARGET_COL}

# Decisions the analyst set; easy to tighten later.
MATURITY_CUTOFF = "2016-12"   # drop loans issued after Dec 2016 (2017 vintages
                              # are ~59% still Current -> too censored to trust)
OOT_SPLIT_DATE = "2015-12"    # train: issued <= this; test: after, up to cutoff
                              # -> train = <=2015, test = all of 2016


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def feature_columns() -> list[str]:
    """Columns marked keep in the triage, minus non-feature keeps."""
    dd = pd.read_csv(DICT_PATH)
    keep = dd.loc[
        dd.keep_decision.astype(str).str.strip().str.lower() == "keep", "column"
    ].tolist()
    return [c for c in keep if c not in NON_FEATURE_KEEPS]


def load_frame(con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Read only the columns we need (features + split/label/derivation cols)."""
    close = False
    if con is None:
        con = sqlite3.connect(DB_PATH)
        close = True
    cols = feature_columns() + sorted(NON_FEATURE_KEEPS) + KFCDT_COLS
    cols = list(dict.fromkeys(cols))  # de-dup, keep order
    quoted = ", ".join(f'"{c}"' for c in cols)
    df = pd.read_sql(f"SELECT {quoted} FROM loans", con)
    if close:
        con.close()
    return df


# --------------------------------------------------------------------------
# Label + filters
# --------------------------------------------------------------------------
def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only finished loans; add binary `default` (1 = bad)."""
    out = df[df[TARGET_COL].isin(FINISHED)].copy()
    out["default"] = out[TARGET_COL].isin(FINISHED_BAD).astype(int)
    return out


_EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def parse_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse messy strings and derive credit_age; drop raw date columns."""
    out = df.copy()

    # term: " 36 months" -> 36
    out["term"] = out["term"].str.extract(r"(\d+)").astype("float")

    # emp_length ordinal
    out["emp_length"] = out["emp_length"].map(_EMP_MAP).astype("float")

    # dates -> credit history length in months (application-time signal)
    issue = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
    ecl = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["credit_age_months"] = (
        (issue.dt.year - ecl.dt.year) * 12 + (issue.dt.month - ecl.dt.month)
    ).clip(lower=0)

    # keep a real datetime for splitting; drop raw feature-invalid columns
    out["issue_dt"] = issue
    out = out.drop(columns=["earliest_cr_line"])
    return out


def apply_maturity_cutoff(df: pd.DataFrame, cutoff: str = MATURITY_CUTOFF) -> pd.DataFrame:
    """Drop loans issued after `cutoff` (YYYY-MM) to limit censoring bias."""
    ceil = pd.Period(cutoff, "M").to_timestamp("M")
    return df[df["issue_dt"] <= ceil].copy()


def time_split(df: pd.DataFrame, split_date: str = OOT_SPLIT_DATE):
    """Out-of-time split: train issued <= split_date, test after."""
    boundary = pd.Period(split_date, "M").to_timestamp("M")
    train = df[df["issue_dt"] <= boundary].copy()
    test = df[df["issue_dt"] > boundary].copy()
    return train, test


# --------------------------------------------------------------------------
# Feature matrices + preprocessing
# --------------------------------------------------------------------------
def split_feature_types(df: pd.DataFrame, feature_cols: list[str]):
    """Partition features into numeric vs categorical.

    Note: pandas >=3.0 reads text as the `str` dtype (not `object`), so we
    key off "is it numeric?" rather than testing for object dtype.
    """
    nums = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in feature_cols if c not in nums]
    return nums, cats


def model_feature_list(df: pd.DataFrame) -> list[str]:
    """Final feature columns after parsing: triage keeps + credit_age,
    minus KFCDT and any leftover non-feature columns."""
    base = [c for c in feature_columns() if c in df.columns and c != "earliest_cr_line"]
    if "credit_age_months" in df.columns and "credit_age_months" not in base:
        base.append("credit_age_months")
    return [c for c in base if c not in KFCDT_COLS]


# --------------------------------------------------------------------------
# LC-model features (KFCDT columns) — for the with/without comparison (3C)
# --------------------------------------------------------------------------
_GRADES = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_ORD = {g: i + 1 for i, g in enumerate(_GRADES)}                 # A..G -> 1..7
SUBGRADE_ORD = {f"{g}{n}": i * 5 + n for i, g in enumerate(_GRADES)   # A1..G5 -> 1..35
                for n in range(1, 6)}


def add_lc_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ordinal-encode LC's grade/sub_grade; int_rate is already numeric.
    Returns the frame plus the list of added LC feature columns."""
    out = df.copy()
    out["grade_ord"] = out["grade"].map(GRADE_ORD)
    out["subgrade_ord"] = out["sub_grade"].map(SUBGRADE_ORD)
    return out, ["grade_ord", "subgrade_ord", "int_rate"]


def build_tree_preprocessor(nums: list[str], cats: list[str], impute: bool):
    """Tree-appropriate preprocessing (no scaling, full one-hot).

    impute=True  -> median-impute numerics (RandomForest can't take NaN).
    impute=False -> pass numerics through with NaN intact (XGBoost/LightGBM
                    handle missing natively, which usually beats imputation).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    num_tf = SimpleImputer(strategy="median") if impute else "passthrough"
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num_tf, nums), ("cat", ohe, cats)])


def build_preprocessor(nums: list[str], cats: list[str]):
    """ColumnTransformer: median-impute + missing-indicator + scale numerics;
    constant-impute + one-hot (drop-first) categoricals. Fit on train only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                 sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", numeric, nums),
        ("cat", categorical, cats),
    ])
