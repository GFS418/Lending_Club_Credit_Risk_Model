"""
Session 3A + 3C — model bake-off with time-aware CV tuning.

3A: RandomForest, XGBoost, LightGBM vs. the logistic baseline, each tuned with
    RandomizedSearchCV over a TimeSeriesSplit (folds respect time), evaluated
    out-of-time on 2016. Model selection is by cross-validated AUC; test metrics
    are the final unbiased report.
3C: refit the winner with vs. without LC's grade/sub_grade/int_rate to measure
    how much signal is genuinely ours vs. re-learning LC's risk grade.

Preprocessing is tree-appropriate: no scaling, full one-hot; XGB/LGBM keep NaN
(native handling), RF gets median imputation. Everything fit inside the pipeline
per CV fold, so no leakage.

Run:
    python src/bakeoff.py            # full tuning (long)
    python src/bakeoff.py --smoke    # tiny subsample to validate the code fast
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import data_prep as dp

REPORTS = dp.ROOT / "reports"
MODELS = dp.ROOT / "models"


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def model_specs(smoke: bool):
    """name -> (estimator, param_distributions, needs_imputation)."""
    if smoke:  # tiny grids just to exercise the code path
        return {
            "RandomForest": (
                RandomForestClassifier(random_state=42, n_jobs=-1),
                {"clf__n_estimators": [80], "clf__max_depth": [8, 12]}, True),
            "XGBoost": (
                XGBClassifier(tree_method="hist", eval_metric="logloss",
                              random_state=42, n_jobs=-1),
                {"clf__n_estimators": [100], "clf__max_depth": [3, 5]}, False),
            "LightGBM": (
                LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
                {"clf__n_estimators": [100], "clf__num_leaves": [31, 63]}, False),
        }
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 12, 20],
                "clf__max_features": ["sqrt", 0.3],
                "clf__min_samples_leaf": [1, 5, 20],
            }, True),
        "XGBoost": (
            XGBClassifier(tree_method="hist", eval_metric="logloss",
                          random_state=42, n_jobs=-1),
            {
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9],
                "clf__colsample_bytree": [0.6, 0.8],
                "clf__min_child_weight": [1, 5],
                "clf__reg_lambda": [1.0, 5.0],
            }, False),
        "LightGBM": (
            LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1,
                           subsample_freq=1),
            {
                "clf__n_estimators": [300, 600],
                "clf__num_leaves": [31, 63, 127],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__colsample_bytree": [0.6, 0.8],
                "clf__subsample": [0.7, 0.9],
                "clf__min_child_samples": [20, 50],
                "clf__reg_lambda": [0.0, 5.0],
            }, False),
    }


def n_iter_for(name: str, smoke: bool) -> int:
    if smoke:
        return 2
    return 10 if name == "RandomForest" else 30  # RF fits are the slowest


def main(smoke: bool) -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # ---- data (reuse Session-2 prep) ------------------------------------
    df = dp.load_frame()
    df = dp.build_label(df)
    df = dp.parse_types(df)
    df = dp.apply_maturity_cutoff(df)
    df, lc_cols = dp.add_lc_features(df)
    train, test = dp.time_split(df)
    train = train.sort_values("issue_dt").reset_index(drop=True)  # for TimeSeriesSplit

    app_feats = dp.model_feature_list(df)
    nums, cats = dp.split_feature_types(df, app_feats)
    empty = [c for c in nums if train[c].notna().sum() == 0]
    nums = [c for c in nums if c not in empty]
    app_feats = [c for c in app_feats if c not in empty]

    if smoke:
        train = train.tail(30_000).reset_index(drop=True)
        test = test.head(10_000)

    y_train, y_test = train["default"], test["default"]
    print(f"train {len(train):,} / test {len(test):,} | "
          f"app features: {len(app_feats)} ({len(nums)} num, {len(cats)} cat) | "
          f"LC add-ons: {lc_cols}")

    tscv = TimeSeriesSplit(n_splits=2 if smoke else 3)
    results = []
    best_params = {}
    fitted = {}  # name -> refit pipeline on app-only

    for name, (est, grid, impute) in model_specs(smoke).items():
        pre = dp.build_tree_preprocessor(nums, cats, impute=impute)
        pipe = Pipeline([("pre", pre), ("clf", est)])
        search = RandomizedSearchCV(
            pipe, grid, n_iter=n_iter_for(name, smoke), scoring="roc_auc",
            cv=tscv, n_jobs=1, refit=True, random_state=42, verbose=1,
        )
        t1 = time.time()
        print(f"\n>>> tuning {name} ...")
        search.fit(train[app_feats], y_train)
        p = search.predict_proba(test[app_feats])[:, 1]
        auc = roc_auc_score(y_test, p)
        rec = {"model": name, "cv_auc": round(search.best_score_, 4),
               "test_auc": round(auc, 4), "test_gini": round(2 * auc - 1, 4),
               "test_ks": round(ks_statistic(y_test, p), 4),
               "fit_seconds": round(time.time() - t1)}
        results.append(rec)
        best_params[name] = {k.replace("clf__", ""): v
                             for k, v in search.best_params_.items()}
        fitted[name] = search.best_estimator_
        print(f"    {name}: CV AUC {rec['cv_auc']} | test AUC {rec['test_auc']} "
              f"({rec['fit_seconds']}s)")

    # ---- pick winner by CROSS-VALIDATED auc (not test) ------------------
    winner = max(results, key=lambda r: r["cv_auc"])["model"]
    print(f"\nWinner (by CV AUC): {winner}")

    # ---- 3C: winner with vs. without LC grade ---------------------------
    lc_feats = app_feats + lc_cols
    nums_lc = nums + lc_cols  # ordinal grade/subgrade + int_rate are numeric
    impute_w = model_specs(smoke)[winner][2]
    win_est = fitted[winner].named_steps["clf"]  # tuned estimator (best params)

    pre_lc = dp.build_tree_preprocessor(nums_lc, cats, impute=impute_w)
    from sklearn.base import clone
    pipe_lc = Pipeline([("pre", pre_lc), ("clf", clone(win_est))])
    pipe_lc.fit(train[lc_feats], y_train)
    p_lc = pipe_lc.predict_proba(test[lc_feats])[:, 1]
    auc_app = next(r for r in results if r["model"] == winner)["test_auc"]
    auc_lc = roc_auc_score(y_test, p_lc)
    comparison = {
        "winner": winner,
        "app_only_test_auc": round(auc_app, 4),
        "app_plus_lc_test_auc": round(auc_lc, 4),
        "auc_lift_from_lc": round(auc_lc - auc_app, 4),
    }
    print(f"\n3C — {winner} app-only AUC {auc_app} vs +LC grade AUC "
          f"{round(auc_lc,4)}  (lift {comparison['auc_lift_from_lc']:+.4f})")

    # ---- persist --------------------------------------------------------
    import pandas as pd
    res_df = pd.DataFrame(results).sort_values("cv_auc", ascending=False)
    suffix = "_smoke" if smoke else ""
    res_df.to_csv(REPORTS / f"bakeoff_results{suffix}.csv", index=False)
    (REPORTS / f"bakeoff_best_params{suffix}.json").write_text(
        json.dumps({"best_params": best_params, "comparison_3c": comparison}, indent=2))
    if not smoke:
        joblib.dump(fitted[winner], MODELS / "winner_app_only.joblib")
        joblib.dump(pipe_lc, MODELS / "winner_with_lc.joblib")

    print("\n" + res_df.to_string(index=False))
    print(f"\nTotal {time.time() - t0:.0f}s. Wrote reports/bakeoff_results{suffix}.csv")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(dp.ROOT / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
