"""
Sensitivity check — stricter maturity cutoff (drop 2016+).

Re-runs the key conclusions with maturity cutoff Dec-2015 (train <=2014,
test 2015; ~10% censored vs 31% for 2016), REUSING the tuned hyperparameters
so we isolate the effect of the cutoff, not the tuning. Compares against the
original Dec-2016 results and prints whether each conclusion holds.

Run:  python src/sensitivity_cutoff.py
"""
from __future__ import annotations

import json
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import data_prep as dp

REPORTS = dp.ROOT / "reports"
ECON = ["total_pymnt", "total_rec_prncp", "recoveries"]
CUTOFF, SPLIT, CALIB_FIT = "2015-12", "2014-12", "2013-12"

# original Dec-2016 results, for side-by-side
ORIG = {"logit": 0.7092, "xgb": 0.7199, "lgbm": 0.7182, "rf": 0.7076,
        "lc_lift": 0.0046, "lgd": 0.633, "approve_all_M": 10.9,
        "pm_threshold": 0.232, "pm_approve": 0.692, "pm_default": 0.158,
        "pm_profit_M": 109.4}


def load_with_econ():
    cols = (dp.feature_columns() + sorted(dp.NON_FEATURE_KEEPS)
            + dp.KFCDT_COLS + ECON)
    cols = list(dict.fromkeys(cols))
    con = sqlite3.connect(dp.DB_PATH)
    df = pd.read_sql(f"SELECT {', '.join(f'\"{c}\"' for c in cols)} FROM loans", con)
    con.close()
    return df


def main():
    df = load_with_econ()
    df = dp.build_label(df)
    df = dp.parse_types(df)
    df = dp.apply_maturity_cutoff(df, cutoff=CUTOFF)
    df, lc_cols = dp.add_lc_features(df)
    train, test = dp.time_split(df, split_date=SPLIT)     # train<=2014, test 2015

    feats = dp.model_feature_list(df)
    nums, cats = dp.split_feature_types(df, feats)
    empty = [c for c in nums if train[c].notna().sum() == 0]
    nums = [c for c in nums if c not in empty]
    feats = [c for c in feats if c not in empty]
    yq = "default"
    print(f"cutoff {CUTOFF}: train {len(train):,} (<=2014) / test {len(test):,} (2015)"
          f" | test default rate {test[yq].mean():.1%}")

    best = json.loads((REPORTS / "bakeoff_best_params.json").read_text())["best_params"]
    Xtr, ytr, Xte, yte = train[feats], train[yq], test[feats], test[yq]
    out = {}

    def evaluate(pipe, X_tr, y_tr, X_te, name):
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(yte, p)
        print(f"  {name:12} AUC {auc:.4f}")
        return auc, pipe

    # logistic baseline
    out["logit"], _ = evaluate(
        Pipeline([("pre", dp.build_preprocessor(nums, cats)),
                  ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))]),
        Xtr, ytr, Xte, "Logistic")
    # tuned trees (reuse params)
    out["rf"], _ = evaluate(
        Pipeline([("pre", dp.build_tree_preprocessor(nums, cats, True)),
                  ("clf", RandomForestClassifier(random_state=42, n_jobs=-1,
                                                 **best["RandomForest"]))]),
        Xtr, ytr, Xte, "RandomForest")
    out["lgbm"], _ = evaluate(
        Pipeline([("pre", dp.build_tree_preprocessor(nums, cats, False)),
                  ("clf", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1,
                                         subsample_freq=1, **best["LightGBM"]))]),
        Xtr, ytr, Xte, "LightGBM")
    out["xgb"], xgb_pipe = evaluate(
        Pipeline([("pre", dp.build_tree_preprocessor(nums, cats, False)),
                  ("clf", XGBClassifier(tree_method="hist", eval_metric="logloss",
                                        random_state=42, n_jobs=-1, **best["XGBoost"]))]),
        Xtr, ytr, Xte, "XGBoost")

    # 3C: with LC grade
    nums_lc = nums + lc_cols
    xgb_lc = Pipeline([("pre", dp.build_tree_preprocessor(nums_lc, cats, False)),
                       ("clf", XGBClassifier(tree_method="hist", eval_metric="logloss",
                                             random_state=42, n_jobs=-1, **best["XGBoost"]))])
    xgb_lc.fit(train[feats + lc_cols], ytr)
    out["lc_lift"] = roc_auc_score(yte, xgb_lc.predict_proba(test[feats + lc_cols])[:, 1]) - out["xgb"]
    print(f"  LC-grade lift +{out['lc_lift']:.4f}")

    # 3B: calibration (fit <=2013, isotonic on 2014, test 2015)
    b = pd.Period(CALIB_FIT, "M").to_timestamp("M")
    fit_df, calib_df = train[train.issue_dt <= b], train[train.issue_dt > b]
    cal_pipe = Pipeline([("pre", dp.build_tree_preprocessor(nums, cats, False)),
                         ("clf", XGBClassifier(tree_method="hist", eval_metric="logloss",
                                               random_state=42, n_jobs=-1, **best["XGBoost"]))])
    cal_pipe.fit(fit_df[feats], fit_df[yq])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_pipe.predict_proba(calib_df[feats])[:, 1], calib_df[yq].to_numpy())
    p_raw = cal_pipe.predict_proba(Xte)[:, 1]
    p_cal = iso.predict(p_raw)
    out["brier_raw"] = brier_score_loss(yte, p_raw)
    out["brier_cal"] = brier_score_loss(yte, p_cal)

    # 3D: P&L on 2015
    econ = test.copy()
    econ["net"] = econ["total_pymnt"].fillna(0) + econ["recoveries"].fillna(0) - econ["funded_amnt"]
    bad = econ[econ[yq] == 1]
    out["lgd"] = float((1 - (bad.total_rec_prncp.fillna(0) + bad.recoveries.fillna(0))
                        / bad.funded_amnt).clip(0, 1).mean())
    approve_all = econ["net"].sum()
    rows = []
    for t in np.linspace(0.02, 0.9, 200):
        appr = p_cal <= t
        if appr.sum() == 0:
            continue
        rows.append({"t": t, "approve": appr.mean(),
                     "book_default": econ[yq][appr].mean(),
                     "profit": econ["net"][appr].sum()})
    sweep = pd.DataFrame(rows)
    pm = sweep.loc[sweep.profit.idxmax()]
    out.update({"approve_all_M": approve_all / 1e6, "pm_threshold": float(pm.t),
                "pm_approve": float(pm.approve), "pm_default": float(pm.book_default),
                "pm_profit_M": pm.profit / 1e6,
                "test_default_rate": float(test[yq].mean())})

    # ---- comparison print ----------------------------------------------
    print("\n" + "=" * 64)
    print(f"{'metric':26} {'Dec-2016 (orig)':>17} {'Dec-2015 (strict)':>18}")
    print("-" * 64)
    def line(k, label, fmt):
        print(f"{label:26} {fmt(ORIG[k]):>17} {fmt(out[k]):>18}")
    line("logit", "Logistic AUC", lambda v: f"{v:.4f}")
    line("rf", "RandomForest AUC", lambda v: f"{v:.4f}")
    line("lgbm", "LightGBM AUC", lambda v: f"{v:.4f}")
    line("xgb", "XGBoost AUC (winner)", lambda v: f"{v:.4f}")
    line("lc_lift", "LC-grade AUC lift", lambda v: f"+{v:.4f}")
    line("lgd", "empirical LGD", lambda v: f"{v:.1%}")
    line("approve_all_M", "approve-all profit", lambda v: f"${v:.1f}M")
    line("pm_threshold", "profit-max PD cutoff", lambda v: f"{v:.3f}")
    line("pm_approve", "  approve rate", lambda v: f"{v:.1%}")
    line("pm_default", "  book default", lambda v: f"{v:.1%}")
    line("pm_profit_M", "  total profit", lambda v: f"${v:.1f}M")
    uplift_orig = ORIG["pm_profit_M"] - ORIG["approve_all_M"]
    uplift_new = out["pm_profit_M"] - out["approve_all_M"]
    print(f"{'profit uplift':26} {f'+${uplift_orig:.1f}M':>17} {f'+${uplift_new:.1f}M':>18}")
    print("=" * 64)

    (REPORTS / "sensitivity_cutoff.json").write_text(json.dumps(
        {"original_2016": ORIG, "strict_2015": out}, indent=2, default=float))

    # P&L figure
    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    ax1.plot(sweep.t, sweep.profit / 1e6, color="#8e44ad", label="profit ($M), 2015 test")
    ax1.axhline(approve_all / 1e6, ls="--", color="#9aa0a6",
                label=f"approve-all (${approve_all/1e6:.1f}M)")
    ax1.axvline(pm.t, ls=":", color="#8e44ad")
    ax1.set(xlabel="approve if P(default) <= threshold",
            ylabel="total realized profit ($M)",
            title="Sensitivity: P&L on 2015 (stricter cutoff)")
    ax1.legend(loc="lower center")
    fig.tight_layout(); fig.savefig(REPORTS / "sensitivity_pnl_2015.png", dpi=120,
                                    bbox_inches="tight")
    print("wrote reports/sensitivity_cutoff.json, sensitivity_pnl_2015.png")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(dp.ROOT / "src"))
    main()
