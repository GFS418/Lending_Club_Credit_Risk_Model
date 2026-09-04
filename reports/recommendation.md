# Credit-Risk Model — Recommendation

**Model → decision → dollars, for the Lending Club accepted-loan book.**

## Summary

We built an application-time credit-default model and evaluated a risk-based
approve/decline policy on an out-of-time 2016 book. **Recommendation: adopt a
probability-of-default (PD) cutoff policy.** Declining the riskiest tail of
applications materially improves realized profit and cuts the book's default
rate, with a fully interpretable and leakage-free model behind each decision.

## The model

- **What it predicts:** probability a loan is charged off, using **only
  information available at application** (no post-origination leakage — the
  discipline was verified end-to-end, including via SHAP).
- **Chosen model:** XGBoost, tuned with time-aware cross-validation, isotonic-
  calibrated. **Out-of-time (2016) performance: ROC-AUC ≈ 0.72, Gini ≈ 0.44,
  KS ≈ 0.32** — a modest but real gain over a logistic baseline (0.71), as
  expected for a largely monotonic credit problem.
- **Independent of LendingClub's own grade:** adding LC's `grade`/`int_rate`
  lifts AUC only **+0.005** — the model reproduces LC's risk assessment from raw
  application data rather than depending on it.

## The decision & the dollars

Backtested on 2016 using **realized cashflows** (no assumed interest;
`total_pymnt + recoveries − funded_amnt` per loan; empirical LGD ≈ 63%):

| Policy | Approve rate | Book default rate | Realized profit |
|---|---|---|---|
| Approve all (current book) | 100% | 23.3% | ~\$11M |
| **PD-cutoff (profit-max, PD ≤ 0.23)** | **69%** | **15.8%** | **~\$109M** |

The profit curve is an inverted-U: declining too little keeps the defaulters,
declining too much forgoes good-loan interest. The optimum declines the riskiest
~31% of applications on this censored 2016 book — **but see limitation 1: on
mature data the honest optimum is a much gentler ~10% decline for ~+\$35M.**

## Limitations & model risk (read before deploying)

1. **Absolute uplift is optimistic — quantified by a sensitivity check.** The
   2016 test book is right-censored (recent vintages over-represent
   fast-defaulters), inflating its default rate and the dollar uplift.
   Re-running on a stricter Dec-2015 cutoff (~10% censored vs 31%) confirms it:
   on the cleaner, genuinely-profitable book the risk-based cutoff still helps,
   but the realistic gain is **≈ +\$35M at a ~10% decline rate**, not +\$98M at
   a 31% decline. **Use ~+\$35M / ~10%-decline as the honest estimate**; the
   *direction* (a PD cutoff raises profit) is robust, but the 2016 magnitude and
   the aggressive cutoff were a censoring artifact.
2. **Calibration drift.** The model under-predicts 2016 default risk (concept
   drift the out-of-time design exposed). Recalibrate on the most recent data
   before and during production use.
3. **Undiscounted.** The P&L ignores time-value of money; discounting compresses
   the profit side.
4. **Commercial aggressiveness.** A 31% decline rate is a large volume cut; the
   business would likely choose a gentler operating point off the profit curve
   (`reports/pnl_sweep.csv`).
5. **Fair lending.** Geography (ZIP/state) was deliberately excluded to avoid a
   protected-class proxy. Any deployment needs a formal disparate-impact review.

## Why it's trustworthy

SHAP confirms the drivers are economically sensible (loan term, payment burden,
FICO, DTI, income, recent credit-seeking) with correct directions, and the
ranking is free of leakage and geographic proxies. Every score decomposes into
intelligible per-feature contributions — suitable for adverse-action reasons and
model-risk review (SR 11-7).

## Recommended next steps

1. Choose an operating cutoff from the profit curve balancing profit vs. volume.
2. Stand up continuous recalibration on recent vintages.
3. Run a formal fair-lending / disparate-impact analysis.
4. Backtest on fully-matured vintages to de-bias the dollar estimates.
