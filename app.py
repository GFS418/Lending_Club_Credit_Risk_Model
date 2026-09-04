"""
Lending Club credit-risk scorer — Session 5 capstone.

Enter a loan application -> calibrated probability of default -> approve/decline
at your chosen cutoff, with a per-loan SHAP explanation of the score.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

BUNDLE = Path(__file__).resolve().parent / "models" / "serving_bundle.joblib"

st.set_page_config(page_title="Lending Club Credit-Risk Scorer", layout="wide")


@st.cache_resource
def load_bundle():
    b = joblib.load(BUNDLE)
    explainer = shap.TreeExplainer(b["pipeline"].named_steps["clf"])
    return b, explainer


def installment(loan_amnt: float, term: int, apr: float) -> float:
    r = apr / 100 / 12
    if r == 0:
        return loan_amnt / term
    return loan_amnt * r / (1 - (1 + r) ** (-term))


if not BUNDLE.exists():
    st.error("Serving model not found. Run `python src/build_serving_model.py` first.")
    st.stop()

bundle, explainer = load_bundle()
pipe, iso = bundle["pipeline"], bundle["calibrator"]
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
LGD = bundle["lgd"]

st.title("Lending Club — Credit-Risk Scorer")
st.caption("Application-time model → calibrated probability of default → "
           "approve/decline decision, with a SHAP explanation. Portfolio demo.")

# --------------------------------------------------------------------------
# Inputs (the top-SHAP features; everything else uses training defaults)
# --------------------------------------------------------------------------
opts = bundle["categorical_options"]
with st.sidebar:
    st.header("Loan application")
    loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 15000, step=500)
    term = st.selectbox("Term (months)", [36, 60], index=0)
    apr = st.slider("Interest rate / APR (%)", 5.0, 31.0, 13.0, 0.5,
                    help="Sets the monthly installment (LC prices this from grade).")
    annual_inc = st.number_input("Annual income ($)", 5000, 500000, 65000, step=1000)
    dti = st.slider("Debt-to-income ratio", 0.0, 45.0, 18.0, 0.5)
    fico = st.slider("FICO score", 660, 850, 700, 5)
    emp_length = st.selectbox("Employment length (yrs)", list(range(0, 11)), index=5)
    home = st.selectbox("Home ownership", opts.get("home_ownership", ["RENT"]))
    purpose = st.selectbox("Loan purpose", opts.get("purpose", ["debt_consolidation"]))
    revol_util = st.slider("Revolving utilization (%)", 0.0, 120.0, 45.0, 1.0)
    open_acc = st.number_input("Open credit lines", 0, 60, 11)
    acc_24m = st.number_input("Accounts opened last 24 mo", 0, 40, 4)
    inq_6m = st.number_input("Credit inquiries last 6 mo", 0, 20, 1)
    cutoff = st.slider("Decline if P(default) exceeds", 0.05, 0.60,
                       bundle["profit_max_threshold"], 0.01,
                       help="Profit-maximizing cutoff from the 3D backtest ≈ 0.23.")

# ---- assemble a full feature row: defaults, then overrides ----------------
row = dict(bundle["defaults"])
inst = installment(loan_amnt, term, apr)
row.update({
    "loan_amnt": loan_amnt, "funded_amnt": loan_amnt, "funded_amnt_inv": loan_amnt,
    "term": float(term), "installment": inst, "annual_inc": annual_inc, "dti": dti,
    "fico_range_high": float(fico), "fico_range_low": float(fico - 4),
    "emp_length": float(emp_length), "home_ownership": home, "purpose": purpose,
    "revol_util": revol_util, "open_acc": open_acc,
    "acc_open_past_24mths": acc_24m, "inq_last_6mths": inq_6m,
})
X = pd.DataFrame([row])[bundle["features"]]

# ---- score ----------------------------------------------------------------
raw = float(pipe.predict_proba(X)[:, 1][0])
pd_cal = float(iso.predict([raw])[0])
exp_loss = pd_cal * LGD * loan_amnt
approve = pd_cal <= cutoff

left, right = st.columns([1, 1.3])
with left:
    st.subheader("Decision")
    if approve:
        st.success("APPROVE")
    else:
        st.error("DECLINE")
    st.metric("Calibrated P(default)", f"{pd_cal:.1%}")
    st.metric("Expected loss (PD × LGD × amount)", f"${exp_loss:,.0f}",
              help=f"LGD ≈ {LGD:.0%} (empirical, from the 2016 backtest).")
    st.metric("Monthly installment", f"${inst:,.0f}")
    st.caption(f"Cutoff {cutoff:.0%}. Model: calibrated XGBoost, application-time "
               "features only (no LC grade). Out-of-time test AUC ≈ 0.72.")

# ---- per-loan SHAP explanation --------------------------------------------
with right:
    st.subheader("Why this score?")
    Xt = pre.transform(X)
    Xt_df = pd.DataFrame(Xt, columns=list(pre.get_feature_names_out()))
    exp = explainer(Xt_df)
    shap.plots.waterfall(exp[0], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    st.pyplot(fig, clear_figure=True)
    st.caption("SHAP contributions (log-odds) from the base rate to this score. "
               "Red pushes risk up, blue pushes it down.")
