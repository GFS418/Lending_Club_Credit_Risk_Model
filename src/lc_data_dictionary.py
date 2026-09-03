"""
Curated reference for Lending Club columns: a short description and a
leakage flag for each well-known field.

Flags (the whole point of the triage):
  TARGET   -> the outcome we predict (loan_status), not a feature
  SAFE     -> known at APPLICATION time; legitimate predictor
  LC_MODEL -> known at application time BUT is Lending Club's own risk
              output (grade / sub_grade / int_rate). Not leakage, but using
              it means partly re-learning LC's model. Use deliberately.
  LEAKAGE  -> only known AFTER the loan is originated / while it is being
              repaid. Using it inflates performance and is invalid. EXCLUDE.
  REVIEW   -> judgment call; decide during triage.

These flags/descriptions are a starting point from the standard LC data
dictionary and domain knowledge -- verify against the empirical stats
(null %, cardinality, sample values) the notebook prints. Columns not
listed here default to REVIEW.
"""
from __future__ import annotations

# col -> (leakage_flag, description)
ANNOTATIONS: dict[str, tuple[str, str]] = {
    # ---- Target -------------------------------------------------------
    "loan_status": ("TARGET", "Current status of the loan; source of the default label."),

    # ---- Loan terms known at origination ------------------------------
    "loan_amnt": ("SAFE", "Requested/listed loan amount."),
    "funded_amnt": ("SAFE", "Amount funded (set at origination)."),
    "funded_amnt_inv": ("SAFE", "Portion funded by investors (set at origination)."),
    "term": ("SAFE", "Number of payments: 36 or 60 months."),
    "installment": ("SAFE", "Monthly payment owed if the loan originates."),
    "issue_d": ("SAFE", "Month the loan was funded (origination date)."),
    "purpose": ("SAFE", "Borrower-stated purpose of the loan."),
    "title": ("REVIEW", "Free-text loan title (borrower-supplied)."),
    "application_type": ("SAFE", "Individual vs Joint application."),
    "disbursement_method": ("SAFE", "Cash vs DirectPay disbursement."),
    "initial_list_status": ("SAFE", "Initial listing status (w/f)."),
    "policy_code": ("REVIEW", "1 = publicly available policy; near-constant."),

    # ---- LC's own risk model (use deliberately) -----------------------
    "grade": ("LC_MODEL", "LC-assigned letter grade (A-G)."),
    "sub_grade": ("LC_MODEL", "LC-assigned sub-grade (A1-G5)."),
    "int_rate": ("LC_MODEL", "Interest rate LC set for the loan."),

    # ---- Borrower attributes at application ---------------------------
    "emp_title": ("REVIEW", "Employer/job title (free text, high cardinality)."),
    "emp_length": ("SAFE", "Employment length in years (0-10+)."),
    "home_ownership": ("SAFE", "RENT/OWN/MORTGAGE/OTHER."),
    "annual_inc": ("SAFE", "Self-reported annual income."),
    "annual_inc_joint": ("SAFE", "Combined income for joint applications."),
    "verification_status": ("SAFE", "Whether income was LC-verified."),
    "verification_status_joint": ("SAFE", "Joint income verification status."),
    "zip_code": ("SAFE", "First 3 digits of borrower ZIP."),
    "addr_state": ("SAFE", "Borrower state."),
    "dti": ("SAFE", "Debt-to-income ratio (ex-mortgage)."),
    "dti_joint": ("SAFE", "Joint debt-to-income ratio."),

    # ---- Credit-bureau attributes at application ----------------------
    "earliest_cr_line": ("SAFE", "Month the earliest credit line was opened."),
    "fico_range_low": ("SAFE", "Lower bound of FICO at application."),
    "fico_range_high": ("SAFE", "Upper bound of FICO at application."),
    "open_acc": ("SAFE", "Number of open credit lines."),
    "total_acc": ("SAFE", "Total number of credit lines ever."),
    "revol_bal": ("SAFE", "Total revolving balance."),
    "revol_util": ("SAFE", "Revolving line utilization rate."),
    "delinq_2yrs": ("SAFE", "30+ day delinquencies in past 2 years."),
    "delinq_amnt": ("SAFE", "Past-due amount owed on delinquent accounts."),
    "inq_last_6mths": ("SAFE", "Credit inquiries in last 6 months."),
    "mths_since_last_delinq": ("SAFE", "Months since last delinquency."),
    "mths_since_last_record": ("SAFE", "Months since last public record."),
    "mths_since_last_major_derog": ("SAFE", "Months since 90-day+ rating."),
    "pub_rec": ("SAFE", "Number of derogatory public records."),
    "pub_rec_bankruptcies": ("SAFE", "Number of public-record bankruptcies."),
    "tax_liens": ("SAFE", "Number of tax liens."),
    "collections_12_mths_ex_med": ("SAFE", "Collections in 12 mths (ex-medical)."),
    "acc_now_delinq": ("SAFE", "Accounts currently delinquent."),
    "chargeoff_within_12_mths": ("SAFE", "Charge-offs within 12 months."),
    "mort_acc": ("SAFE", "Number of mortgage accounts."),
    "open_acc_6m": ("SAFE", "Open trades in last 6 months."),
    "tot_coll_amt": ("SAFE", "Total collection amounts ever owed."),
    "tot_cur_bal": ("SAFE", "Total current balance of all accounts."),
    "total_rev_hi_lim": ("SAFE", "Total revolving high credit/limit."),

    # ---- POST-ORIGINATION: LEAKAGE. EXCLUDE. --------------------------
    "out_prncp": ("LEAKAGE", "Remaining outstanding principal."),
    "out_prncp_inv": ("LEAKAGE", "Remaining outstanding principal (investor)."),
    "total_pymnt": ("LEAKAGE", "Payments received to date."),
    "total_pymnt_inv": ("LEAKAGE", "Payments received to date (investor)."),
    "total_rec_prncp": ("LEAKAGE", "Principal received to date."),
    "total_rec_int": ("LEAKAGE", "Interest received to date."),
    "total_rec_late_fee": ("LEAKAGE", "Late fees received to date."),
    "recoveries": ("LEAKAGE", "Post charge-off gross recovery."),
    "collection_recovery_fee": ("LEAKAGE", "Post charge-off collection fee."),
    "last_pymnt_d": ("LEAKAGE", "Date of last payment received."),
    "last_pymnt_amnt": ("LEAKAGE", "Amount of last payment received."),
    "next_pymnt_d": ("LEAKAGE", "Next scheduled payment date."),
    "last_credit_pull_d": ("LEAKAGE", "Most recent credit pull (post-origination)."),
    "last_fico_range_high": ("LEAKAGE", "Most recent FICO (updated after origination)."),
    "last_fico_range_low": ("LEAKAGE", "Most recent FICO (updated after origination)."),
    "pymnt_plan": ("LEAKAGE", "On a payment plan (post-origination flag)."),
    "hardship_flag": ("LEAKAGE", "Hardship plan flag (post-origination)."),
    "debt_settlement_flag": ("LEAKAGE", "Debt settlement flag (post-origination)."),
    "debt_settlement_flag_date": ("LEAKAGE", "Date debt-settlement flag set (post-origination)."),
    "payment_plan_start_date": ("LEAKAGE", "Payment-plan start date (post-origination)."),
    "orig_projected_additional_accrued_interest": (
        "LEAKAGE", "Projected accrued interest under hardship (post-origination)."),

    # ---- REVIEW: descriptions from the official LC data dictionary -----
    # Kept as REVIEW so you make the keep/drop call; meanings filled in.
    # Most are application-time credit-bureau attributes (reasonable keeps);
    # the sec_app_* / *_joint ones only populate for joint applications;
    # id / member_id / url / desc / title are identifiers or free text.
    "id": ("REVIEW", "Unique LC id for the loan listing (identifier, not a feature)."),
    "member_id": ("REVIEW", "Deprecated member id; blank in this release."),
    "url": ("REVIEW", "URL of the LC listing page (contains the loan id)."),
    "desc": ("REVIEW", "Free-text loan description written by the borrower."),
    "acc_open_past_24mths": ("REVIEW", "Trades opened in the past 24 months."),
    "avg_cur_bal": ("REVIEW", "Average current balance across all accounts."),
    "bc_open_to_buy": ("REVIEW", "Total open-to-buy on revolving bankcards."),
    "bc_util": ("REVIEW", "Balance-to-limit ratio across bankcard accounts."),
    "all_util": ("REVIEW", "Balance-to-limit ratio across all trades."),
    "il_util": ("REVIEW", "Balance-to-limit ratio across installment accounts."),
    "inq_fi": ("REVIEW", "Number of personal-finance inquiries."),
    "inq_last_12m": ("REVIEW", "Credit inquiries in the past 12 months."),
    "max_bal_bc": ("REVIEW", "Maximum current balance on any revolving account."),
    "mo_sin_old_il_acct": ("REVIEW", "Months since oldest installment account opened."),
    "mo_sin_old_rev_tl_op": ("REVIEW", "Months since oldest revolving account opened."),
    "mo_sin_rcnt_rev_tl_op": ("REVIEW", "Months since most recent revolving account opened."),
    "mo_sin_rcnt_tl": ("REVIEW", "Months since most recent account opened."),
    "mths_since_rcnt_il": ("REVIEW", "Months since most recent installment account opened."),
    "mths_since_recent_bc": ("REVIEW", "Months since most recent bankcard account opened."),
    "mths_since_recent_bc_dlq": ("REVIEW", "Months since most recent bankcard delinquency."),
    "mths_since_recent_inq": ("REVIEW", "Months since most recent credit inquiry."),
    "mths_since_recent_revol_delinq": ("REVIEW", "Months since most recent revolving delinquency."),
    "num_accts_ever_120_pd": ("REVIEW", "Accounts ever 120+ days past due."),
    "num_actv_bc_tl": ("REVIEW", "Currently active bankcard accounts."),
    "num_actv_rev_tl": ("REVIEW", "Currently active revolving trades."),
    "num_bc_sats": ("REVIEW", "Satisfactory bankcard accounts."),
    "num_bc_tl": ("REVIEW", "Number of bankcard accounts."),
    "num_il_tl": ("REVIEW", "Number of installment accounts."),
    "num_op_rev_tl": ("REVIEW", "Number of open revolving accounts."),
    "num_rev_accts": ("REVIEW", "Number of revolving accounts."),
    "num_rev_tl_bal_gt_0": ("REVIEW", "Revolving trades with a balance > 0."),
    "num_sats": ("REVIEW", "Number of satisfactory accounts."),
    "num_tl_120dpd_2m": ("REVIEW", "Accounts 120 days past due (updated in last 2 months)."),
    "num_tl_30dpd": ("REVIEW", "Accounts 30 days past due (updated in last 2 months)."),
    "num_tl_90g_dpd_24m": ("REVIEW", "Accounts 90+ days past due in last 24 months."),
    "num_tl_op_past_12m": ("REVIEW", "Accounts opened in the past 12 months."),
    "open_act_il": ("REVIEW", "Currently active installment trades."),
    "open_il_12m": ("REVIEW", "Installment accounts opened in past 12 months."),
    "open_il_24m": ("REVIEW", "Installment accounts opened in past 24 months."),
    "open_rv_12m": ("REVIEW", "Revolving trades opened in past 12 months."),
    "open_rv_24m": ("REVIEW", "Revolving trades opened in past 24 months."),
    "pct_tl_nvr_dlq": ("REVIEW", "Percent of trades never delinquent."),
    "percent_bc_gt_75": ("REVIEW", "Percent of bankcards over 75% of their limit."),
    "tot_hi_cred_lim": ("REVIEW", "Total high credit / credit limit."),
    "total_bal_ex_mort": ("REVIEW", "Total credit balance excluding mortgage."),
    "total_bal_il": ("REVIEW", "Total current balance of all installment accounts."),
    "total_bc_limit": ("REVIEW", "Total bankcard high credit / credit limit."),
    "total_cu_tl": ("REVIEW", "Number of finance (credit-union) trades."),
    "total_il_high_credit_limit": ("REVIEW", "Total installment high credit / credit limit."),
    "revol_bal_joint": ("REVIEW", "Combined revolving balance of co-borrowers (joint apps)."),
    "sec_app_earliest_cr_line": ("REVIEW", "Secondary applicant: earliest credit line (joint apps)."),
    "sec_app_fico_range_low": ("REVIEW", "Secondary applicant: FICO lower bound (joint apps)."),
    "sec_app_fico_range_high": ("REVIEW", "Secondary applicant: FICO upper bound (joint apps)."),
    "sec_app_inq_last_6mths": ("REVIEW", "Secondary applicant: inquiries in last 6 months (joint apps)."),
    "sec_app_mort_acc": ("REVIEW", "Secondary applicant: mortgage accounts (joint apps)."),
    "sec_app_open_acc": ("REVIEW", "Secondary applicant: open trades (joint apps)."),
    "sec_app_revol_util": ("REVIEW", "Secondary applicant: revolving utilization (joint apps)."),
    "sec_app_open_act_il": ("REVIEW", "Secondary applicant: active installment trades (joint apps)."),
    "sec_app_num_rev_accts": ("REVIEW", "Secondary applicant: revolving accounts (joint apps)."),
    "sec_app_chargeoff_within_12_mths": ("REVIEW", "Secondary applicant: charge-offs in last 12m (joint apps)."),
    "sec_app_collections_12_mths_ex_med": ("REVIEW", "Secondary applicant: collections in last 12m ex-medical (joint apps)."),
    "sec_app_mths_since_last_major_derog": ("REVIEW", "Secondary applicant: months since 90-day+ rating (joint apps)."),
}

# Prefix-based rules for families of post-origination columns (hardship_*,
# settlement_*, sec_app_* etc.). Matched if a column starts with the prefix
# and is not already annotated above.
PREFIX_RULES: list[tuple[str, str, str]] = [
    ("hardship_", "LEAKAGE", "Hardship-program field (post-origination)."),
    ("settlement_", "LEAKAGE", "Debt-settlement field (post-origination)."),
    ("deferral_", "LEAKAGE", "Payment-deferral field (post-origination)."),
]


def annotate(column: str) -> tuple[str, str]:
    """Return (leakage_flag, description) for a column."""
    if column in ANNOTATIONS:
        return ANNOTATIONS[column]
    for prefix, flag, desc in PREFIX_RULES:
        if column.startswith(prefix):
            return flag, desc
    return "REVIEW", ""
