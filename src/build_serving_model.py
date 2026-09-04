"""
Session 5 — build the serving bundle for the Streamlit app.

Persists the calibrated XGBoost (fit <=2014, isotonic on 2015 — the same pair
behind the 3B calibration curve), plus everything the app needs to turn a few
user inputs into a full feature vector: per-feature training defaults, category
options, the feature order, and business constants (LGD, profit-max cutoff).

Run:  python src/build_serving_model.py   ->  models/serving_bundle.joblib
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import data_prep as dp

MODELS = dp.ROOT / "models"
REPORTS = dp.ROOT / "reports"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    df = dp.load_frame()
    df = dp.build_label(df)
    df = dp.parse_types(df)
    df = dp.apply_maturity_cutoff(df)
    train, _ = dp.time_split(df)

    app_feats = dp.model_feature_list(df)
    nums, cats = dp.split_feature_types(df, app_feats)
    empty = [c for c in nums if train[c].notna().sum() == 0]
    nums = [c for c in nums if c not in empty]
    app_feats = [c for c in app_feats if c not in empty]

    boundary = pd.Period("2014-12", "M").to_timestamp("M")
    fit_df, calib_df = train[train.issue_dt <= boundary], train[train.issue_dt > boundary]

    best = json.loads((REPORTS / "bakeoff_best_params.json").read_text())
    xgb_params = best["best_params"]["XGBoost"]
    pipe = Pipeline([
        ("pre", dp.build_tree_preprocessor(nums, cats, impute=False)),
        ("clf", XGBClassifier(tree_method="hist", eval_metric="logloss",
                              random_state=42, n_jobs=-1, **xgb_params)),
    ])
    print("fitting serving model on <=2014 ...")
    pipe.fit(fit_df[app_feats], fit_df["default"])

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pipe.predict_proba(calib_df[app_feats])[:, 1], calib_df["default"].to_numpy())

    # defaults + options so the app can build a full row from partial input
    defaults = {}
    for c in nums:
        defaults[c] = float(fit_df[c].median())
    for c in cats:
        m = fit_df[c].mode(dropna=True)  # empty if the column is all-null in <=2014
        defaults[c] = str(m.iloc[0]) if len(m) else "Missing"
    options = {c: sorted(fit_df[c].dropna().astype(str).unique().tolist()) for c in cats}

    bundle = {
        "pipeline": pipe,
        "calibrator": iso,
        "features": app_feats,
        "numeric_features": nums,
        "categorical_features": cats,
        "defaults": defaults,
        "categorical_options": options,
        "lgd": 0.633,                 # empirical, from 3D
        "profit_max_threshold": 0.232,
        "train_base_rate": float(fit_df["default"].mean()),
    }
    joblib.dump(bundle, MODELS / "serving_bundle.joblib")
    print(f"wrote {MODELS/'serving_bundle.joblib'} "
          f"({len(app_feats)} features, {len(cats)} categorical)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(dp.ROOT / "src"))
    main()
