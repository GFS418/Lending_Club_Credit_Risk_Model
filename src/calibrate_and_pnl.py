"""
Session 3B + 3D — calibrate the winner, then a realized-cashflow P&L backtest.

3B: XGBoost probabilities aren't guaranteed calibrated, and the P&L needs a
    real probability of default. We carve a TIME-BASED calibration fold:
      fit  on loans issued <= 2014
      calibrate (isotonic) on 2015   <- the carved fold
      test on 2016                   <- untouched out-of-time set
    Report Brier score + reliability curve, before vs after calibration.

3D: Backtest the approve/decline policy on 2016 using REALIZED cashflows:
      net_cashflow = total_pymnt + recoveries - funded_amnt   (per loan)
    Sweep the PD cutoff, report approve rate / book default rate / total
    realized profit, and find the profit-maximizing cutoff vs. approve-all.
    (total_pymnt/recoveries are leakage as *features*; used here only to score
    the policy's realized outcome, which the model never sees.)

Run:  python src/calibrate_and_pnl.py
"""
from __future__ import annotations

import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import data_prep as dp

REPORTS = dp.ROOT / "reports"
ECON_COLS = ["total_pymnt", "total_rec_prncp", "recoveries"]


def load_with_econ() -> pd.DataFrame:
    cols = (dp.feature_columns() + sorted(dp.NON_FEATURE_KEEPS)
            + dp.KFCDT_COLS + ECON_COLS)
    cols = list(dict.fromkeys(cols))
    con = sqlite3.connect(dp.DB_PATH)
    quoted = ", ".join(f'"{c}"' for c in cols)
    df = pd.read_sql(f"SELECT {quoted} FROM loans", con)
    con.close()
    return df


def main() -> None:
    # ---- data ----------------------------------------------------------
    df = load_with_econ()
    df = dp.build_label(df)
    df = dp.parse_types(df)
    df = dp.apply_maturity_cutoff(df)
    train, test = dp.time_split(df)

    app_feats = dp.model_feature_list(df)
    nums, cats = dp.split_feature_types(df, app_feats)
    empty = [c for c in nums if train[c].notna().sum() == 0]
    nums = [c for c in nums if c not in empty]
    app_feats = [c for c in app_feats if c not in empty]

    # ---- 3B: time-based calibration fold -------------------------------
    boundary = pd.Period("2014-12", "M").to_timestamp("M")
    fit_mask = train["issue_dt"] <= boundary        # <= 2014
    fit_df, calib_df = train[fit_mask], train[~fit_mask]  # calib = 2015
    print(f"calibration split -> fit {len(fit_df):,} (<=2014) | "
          f"calibrate {len(calib_df):,} (2015) | test {len(test):,} (2016)")

    best = json.loads((REPORTS / "bakeoff_best_params.json").read_text())
    xgb_params = best["best_params"]["XGBoost"]
    pipe = Pipeline([
        ("pre", dp.build_tree_preprocessor(nums, cats, impute=False)),
        ("clf", XGBClassifier(tree_method="hist", eval_metric="logloss",
                              random_state=42, n_jobs=-1, **xgb_params)),
    ])
    print("fitting XGBoost on <=2014 ...")
    pipe.fit(fit_df[app_feats], fit_df["default"])

    # isotonic calibration fit on the 2015 fold's raw predictions
    p_calib_raw = pipe.predict_proba(calib_df[app_feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_calib_raw, calib_df["default"].to_numpy())

    y_test = test["default"].to_numpy()
    p_raw = pipe.predict_proba(test[app_feats])[:, 1]
    p_cal = iso.predict(p_raw)

    brier_raw, brier_cal = brier_score_loss(y_test, p_raw), brier_score_loss(y_test, p_cal)
    print(f"\n3B calibration (2016 test):")
    print(f"  AUC (unchanged by calibration): {roc_auc_score(y_test, p_cal):.4f}")
    print(f"  Brier  raw {brier_raw:.5f}  ->  calibrated {brier_cal:.5f}"
          f"  ({100*(brier_raw-brier_cal)/brier_raw:+.1f}%)")

    # reliability curve
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for p, lab in [(p_raw, f"raw (Brier {brier_raw:.4f})"),
                   (p_cal, f"isotonic (Brier {brier_cal:.4f})")]:
        xf, yf = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(xf, yf, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.set(xlabel="mean predicted P(default)", ylabel="observed default rate",
           title="3B — calibration (2016 out-of-time)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(REPORTS / "calibration_curve.png", dpi=120,
                                    bbox_inches="tight")

    # ---- 3D: realized-cashflow P&L backtest on 2016 --------------------
    econ = test.copy()
    econ["net_cashflow"] = (econ["total_pymnt"].fillna(0) + econ["recoveries"].fillna(0)
                            - econ["funded_amnt"])
    econ["pd"] = p_cal

    # descriptive empirical LGD on defaulted loans
    bad = econ[econ["default"] == 1]
    lgd = (1 - (bad["total_rec_prncp"].fillna(0) + bad["recoveries"].fillna(0))
           / bad["funded_amnt"]).clip(0, 1)
    print(f"\n3D economics:")
    print(f"  empirical LGD on 2016 defaults: mean {lgd.mean():.1%} "
          f"(median {lgd.median():.1%})")

    approve_all_profit = econ["net_cashflow"].sum()
    thresholds = np.linspace(0.02, 0.9, 200)
    rows = []
    for t in thresholds:
        appr = econ["pd"] <= t
        n = int(appr.sum())
        if n == 0:
            continue
        sub = econ[appr]
        rows.append({
            "threshold": t,
            "approve_rate": appr.mean(),
            "book_default_rate": sub["default"].mean(),
            "total_profit": sub["net_cashflow"].sum(),
            "avg_profit_per_loan": sub["net_cashflow"].mean(),
        })
    sweep = pd.DataFrame(rows)
    best_row = sweep.loc[sweep["total_profit"].idxmax()]
    sweep.to_csv(REPORTS / "pnl_sweep.csv", index=False)

    print(f"  approve-all (current book): {len(econ):,} loans, "
          f"total realized profit ${approve_all_profit/1e6:.1f}M, "
          f"default rate {econ['default'].mean():.1%}")
    print(f"  profit-max cutoff PD<={best_row['threshold']:.3f}: "
          f"approve {best_row['approve_rate']:.1%}, "
          f"book default {best_row['book_default_rate']:.1%}, "
          f"total profit ${best_row['total_profit']/1e6:.1f}M "
          f"(+${(best_row['total_profit']-approve_all_profit)/1e6:.1f}M)")

    # P&L figure
    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    ax1.plot(sweep["threshold"], sweep["total_profit"] / 1e6, color="#2e7d32",
             label="total realized profit ($M)")
    ax1.axhline(approve_all_profit / 1e6, ls="--", color="#9aa0a6",
                label=f"approve-all (${approve_all_profit/1e6:.1f}M)")
    ax1.axvline(best_row["threshold"], ls=":", color="#2e7d32")
    ax1.set_xlabel("approve if predicted P(default) <= threshold")
    ax1.set_ylabel("total realized profit on 2016 book ($M)", color="#2e7d32")
    ax2 = ax1.twinx()
    ax2.plot(sweep["threshold"], 100 * sweep["approve_rate"], color="#4a7ebb",
             alpha=0.7, label="approve rate (%)")
    ax2.set_ylabel("approve rate (%)", color="#4a7ebb")
    ax1.set_title("3D — approve/decline policy vs. realized 2016 profit")
    ax1.legend(loc="lower center")
    fig.tight_layout(); fig.savefig(REPORTS / "pnl_threshold.png", dpi=120,
                                    bbox_inches="tight")
    print("\nWrote reports/calibration_curve.png, pnl_threshold.png, pnl_sweep.csv")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(dp.ROOT / "src"))
    main()
