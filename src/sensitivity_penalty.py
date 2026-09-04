"""
Sensitivity check — L1 / elastic-net logistic vs. the L2 baseline.

Follows up the redundancy discussion: does a *sparse* logistic (L1/elastic-net,
which zero out redundant coefficients) match the full L2 model, and how many
features does it keep? Same Dec-2016 setup as the Session-2 baseline (train
<=2015, test 2016), so AUCs are directly comparable to 0.709.

Run:  python src/sensitivity_penalty.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

import data_prep as dp

REPORTS = dp.ROOT / "reports"


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def main():
    df = dp.load_frame()
    df = dp.build_label(df)
    df = dp.parse_types(df)
    df = dp.apply_maturity_cutoff(df)          # Dec-2016 (default)
    train, test = dp.time_split(df)            # train <=2015, test 2016

    feats = dp.model_feature_list(df)
    nums, cats = dp.split_feature_types(df, feats)
    empty = [c for c in nums if train[c].notna().sum() == 0]
    nums = [c for c in nums if c not in empty]
    feats = [c for c in feats if c not in empty]

    Xtr, ytr = train[feats], train["default"]
    Xte, yte = test[feats], test["default"]

    variants = [
        ("L2 (ridge)", dict(penalty="l2")),
        ("L1 (lasso)", dict(penalty="l1")),
        ("elastic-net", dict(penalty="elasticnet", l1_ratio=0.5)),
    ]
    results = []
    for name, kw in variants:
        pre = dp.build_preprocessor(nums, cats)
        clf = LogisticRegression(solver="saga", C=1.0, max_iter=2000, tol=1e-3,
                                 random_state=42, **kw)
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        coef = pipe.named_steps["clf"].coef_[0]
        nz = int((np.abs(coef) > 1e-6).sum())
        auc = roc_auc_score(yte, p)
        rec = {"model": name, "test_auc": round(auc, 4), "test_gini": round(2 * auc - 1, 4),
               "test_ks": round(ks(yte, p), 4), "n_features": len(coef),
               "n_nonzero": nz, "pct_kept": round(100 * nz / len(coef), 1)}
        results.append(rec)
        print(f"  {name:12} AUC {rec['test_auc']} | KS {rec['test_ks']} | "
              f"features kept {nz}/{len(coef)} ({rec['pct_kept']}%)")

    (REPORTS / "sensitivity_penalty.json").write_text(json.dumps(results, indent=2))
    print("\nBaseline (Session 2 L2, lbfgs): AUC 0.7092")
    print("wrote reports/sensitivity_penalty.json")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(dp.ROOT / "src"))
    main()
