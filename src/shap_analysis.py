"""
Session 4 — SHAP interpretability of the XGBoost winner.

Explains the app-only winner (models/winner_app_only.joblib) with TreeSHAP:
  * global: mean |SHAP| bar + beeswarm (which features drive risk, and how)
  * local: waterfall for the highest- and lowest-risk loans in the sample
The goal is a sanity check — the drivers should be economically sensible and
free of proxy/leakage surprises.

Run:  python src/shap_analysis.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib

import data_prep as dp

REPORTS = dp.ROOT / "reports"
MODELS = dp.ROOT / "models"
N_SAMPLE = 10_000


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    pipe = joblib.load(MODELS / "winner_app_only.joblib")
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]

    # ---- rebuild the 2016 test set (same prep as training) -------------
    df = dp.load_frame()
    df = dp.build_label(df)
    df = dp.parse_types(df)
    df = dp.apply_maturity_cutoff(df)
    _, test = dp.time_split(df)

    feat_in = list(pre.feature_names_in_)          # exact columns the model expects
    sample = test.sample(n=min(N_SAMPLE, len(test)), random_state=42)
    X = sample[feat_in]

    # transform to the encoded matrix SHAP will explain, keep feature names
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out())
    Xt_df = pd.DataFrame(Xt, columns=names, index=sample.index)
    print(f"Explaining {len(Xt_df):,} loans x {Xt_df.shape[1]} encoded features")

    # ---- TreeSHAP -------------------------------------------------------
    explainer = shap.TreeExplainer(clf)
    exp = explainer(Xt_df)   # Explanation; values are log-odds contributions

    # ---- global importance table ---------------------------------------
    imp = (pd.DataFrame({"feature": names,
                         "mean_abs_shap": abs(exp.values).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp.to_csv(REPORTS / "shap_importance.csv", index=False)
    print("\nTop 15 features by mean |SHAP|:")
    print(imp.head(15).to_string(index=False))

    # ---- global plots ---------------------------------------------------
    shap.plots.bar(exp, max_display=15, show=False)
    plt.title("SHAP global importance — XGBoost (app-only)")
    plt.tight_layout(); plt.savefig(REPORTS / "shap_bar.png", dpi=120,
                                    bbox_inches="tight"); plt.close("all")

    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("SHAP beeswarm — feature value vs. effect on default risk")
    plt.tight_layout(); plt.savefig(REPORTS / "shap_beeswarm.png", dpi=120,
                                    bbox_inches="tight"); plt.close("all")

    # ---- local explanations: highest- and lowest-risk loans -------------
    proba = clf.predict_proba(Xt)[:, 1]
    hi, lo = proba.argmax(), proba.argmin()
    for pos, tag in [(hi, "highrisk"), (lo, "lowrisk")]:
        shap.plots.waterfall(exp[pos], max_display=12, show=False)
        plt.title(f"Why this loan? predicted P(default)={proba[pos]:.1%} "
                  f"(actual: {'default' if sample['default'].iloc[pos] else 'paid'})")
        plt.tight_layout(); plt.savefig(REPORTS / f"shap_waterfall_{tag}.png",
                                        dpi=120, bbox_inches="tight"); plt.close("all")

    print("\nWrote reports/shap_bar.png, shap_beeswarm.png, "
          "shap_waterfall_highrisk.png, shap_waterfall_lowrisk.png, shap_importance.csv")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(dp.ROOT / "src"))
    main()
