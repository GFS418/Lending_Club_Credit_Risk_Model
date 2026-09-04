"""
Session 2 — logistic-regression baseline for credit default.

Steps:
  A. label (Charged Off/Default = 1, Fully Paid = 0; others dropped)
  B. maturity cutoff (drop loans issued after Feb 2017)
  C. out-of-time split (train issued <= 2015-12, test after)
  D. preprocessing pipeline (fit on train only)
  E. fit L2 logistic; report ROC-AUC, KS, Gini on the out-of-time test set

Run:  python src/train_baseline.py
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

import data_prep as dp


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def main() -> None:
    t0 = time.time()

    # ---- A. load + label -------------------------------------------------
    df = dp.load_frame()
    print(f"Loaded {len(df):,} rows.")
    df = dp.build_label(df)
    print(f"Finished loans: {len(df):,}  |  base default rate "
          f"{df['default'].mean():.3%}")

    # ---- parse types, B. maturity cutoff --------------------------------
    df = dp.parse_types(df)
    before = len(df)
    df = dp.apply_maturity_cutoff(df)
    print(f"After maturity cutoff (<= {dp.MATURITY_CUTOFF}): {len(df):,} "
          f"({before - len(df):,} dropped)  |  default rate {df['default'].mean():.3%}")

    # ---- C. out-of-time split -------------------------------------------
    train, test = dp.time_split(df)
    feats = dp.model_feature_list(df)
    nums, cats = dp.split_feature_types(df, feats)
    print(f"\nOut-of-time split at {dp.OOT_SPLIT_DATE}:")
    print(f"  train: {len(train):,} rows  (issued {train.issue_dt.min():%Y-%m} "
          f"to {train.issue_dt.max():%Y-%m}), default {train['default'].mean():.3%}")
    print(f"  test : {len(test):,} rows  (issued {test.issue_dt.min():%Y-%m} "
          f"to {test.issue_dt.max():%Y-%m}), default {test['default'].mean():.3%}")
    print(f"  features: {len(feats)} ({len(nums)} numeric, {len(cats)} categorical)")

    # Drop features that are entirely missing in TRAIN — they can't be fit
    # (e.g. sec_app_* fields that didn't exist in LC's data before ~2017).
    all_null_in_train = [c for c in nums if train[c].notna().sum() == 0]
    if all_null_in_train:
        print(f"  dropping {len(all_null_in_train)} features empty in train "
              f"(later-vintage only): {', '.join(all_null_in_train)}")
        nums = [c for c in nums if c not in all_null_in_train]
        feats = [c for c in feats if c not in all_null_in_train]

    X_train, y_train = train[feats], train["default"]
    X_test, y_test = test[feats], test["default"]

    # ---- D. preprocessing + E. model (one pipeline) ---------------------
    pre = dp.build_preprocessor(nums, cats)
    # L2 (ridge) logistic. sklearn >=1.8 controls the penalty via C / l1_ratio;
    # defaults give L2, so we just set the strength C.
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    print("\nFitting logistic baseline ...")
    pipe.fit(X_train, y_train)

    # ---- evaluate on the out-of-time test set ---------------------------
    p_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    ks = ks_statistic(y_test, p_test)
    gini = 2 * auc - 1

    # in-sample for reference (overfitting gauge)
    p_train = pipe.predict_proba(X_train)[:, 1]
    auc_tr = roc_auc_score(y_train, p_train)

    print("\n" + "=" * 52)
    print("BASELINE — L2 logistic regression (out-of-time test)")
    print("=" * 52)
    print(f"  ROC-AUC : {auc:.4f}   (train {auc_tr:.4f})")
    print(f"  Gini    : {gini:.4f}")
    print(f"  KS      : {ks:.4f}")
    print(f"  n_features after encoding: "
          f"{pipe.named_steps['pre'].transform(X_test[:1]).shape[1]}")
    print(f"\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(dp.ROOT / "src"))
    main()
