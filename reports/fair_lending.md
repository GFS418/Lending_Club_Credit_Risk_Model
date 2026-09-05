# Fair Lending — Methodology

How this credit model addresses fair-lending risk: the **prevention** already
built into the model, and the **verification** procedure that would confirm it
in production. Credit underwriting is a regulated activity; a predictive model
is only deployable if it is also *defensible*.

## Regulatory context

US fair-lending law (**ECOA / Regulation B**, and the **Fair Housing Act** for
mortgages) recognizes two theories of discrimination:

- **Disparate treatment** — *intentionally* using a protected characteristic
  (race, color, religion, national origin, sex, marital status, age, receipt of
  public assistance). Avoided by simply not using these variables in the model.
- **Disparate impact** — a *facially neutral* model, applied uniformly, that
  nonetheless produces materially worse outcomes for a protected group, without
  adequate **business necessity**, or where a **less discriminatory alternative
  (LDA)** exists. This is the subtle risk: it can occur **even when no protected
  variable is used**, because neutral-looking features act as *proxies* for
  protected status.

The central point: **feature exclusion is prevention; measuring outcomes across
groups is verification.** Both are required — excluding a proxy lowers risk but
does not prove the absence of disparate impact.

## Prevention already applied in this model

1. **No protected attributes** are used as features.
2. **Geography excluded.** `zip_code` and `addr_state` were deliberately dropped
   in the Session-1 triage. Geography is the classic redlining proxy for race;
   removing it eliminates the most notorious source of proxy discrimination.
3. **Interpretability for adverse action.** SHAP produces per-applicant reason
   codes (Session 4), supporting the ECOA requirement to disclose the principal
   reasons for a denial.

These are necessary but **not sufficient** — other features (income,
employment, some bureau variables) can still carry residual proxy signal, which
only the verification step below can detect.

## Verification methodology (disparate-impact analysis)

**1. Obtain protected-class labels.** Mortgage lenders have these via **HMDA**;
consumer lenders typically do not collect race, so it is *inferred* with **BISG**
(Bayesian Improved Surname Geocoding — race probability from surname + location).
These labels are themselves estimates, which adds noise to the analysis.

**2. Disaggregate outcomes by group at a *fixed* cutoff.** Apply the single
approve/decline threshold (e.g. PD ≤ 0.23) to *everyone*, then measure outcomes
*within* each protected group — by race, by sex, and by age bucket. Measure not
only:
- **Approval/decline rates** (*demographic parity*), but also
- **Error rates** — is the false-decline rate (creditworthy applicants wrongly
  declined) higher for one group? (*equalized odds / equal opportunity*), which
  is often the more meaningful harm.

Compare **at similar risk levels** (or via regression that isolates the group
effect after controlling for legitimate risk factors) — a raw approval-rate gap
conflates model bias with genuine differences in measured risk.

*Age note:* under ECOA an empirically-derived scorecard may use age under
conditions but must not penalize applicants **62 or older**; age is the one
protected basis that is not a flat prohibition.

**3. Screen with the four-fifths (80%) rule.** Flag any group whose approval
rate is below 80% of the most-favored group's. Caveats: this is a **screening
heuristic** (from EEOC employment law), not a legal safe harbor — passing it is
not exoneration, and failing it triggers investigation, not automatic liability.
Pair it with a **statistical-significance test**, since a small group can fail
by chance.

**4. Test business necessity, then search for a less discriminatory
alternative.** For any material disparity, ask whether it is driven by a
*genuine* risk factor. A direct, causal risk driver (e.g. high **DTI** predicting
default) is defensible business necessity even if it correlates with a protected
group. **A proxy cannot be the justification** — "applicants in a certain
geography default more" is *not* a valid defense, because geography is a race
proxy; that is the violation, not the excuse. Then search for an LDA — a model
with comparable risk separation and less disparity — via: dropping/transforming
proxy features, **reweighting** training data, adding a **fairness constraint** to
the objective, or **adversarial de-biasing**.

> **Critical pitfall:** do **not** "fix" a gap by setting different score cutoffs
> per protected group. Using the protected class in the decision — even to help a
> group — is **disparate treatment** and is generally impermissible in US
> lending. The remedy must be a genuinely better *neutral* model.

**5. Adverse-action reason codes.** For each denial, surface the applicant's
actual top SHAP drivers, translated to plain language ("high debt-to-income
ratio," "too many recent credit inquiries") — not raw feature names or
coefficients — to satisfy the ECOA principal-reasons requirement.

## Why this is a methodology, not a computed result

The Lending Club dataset **contains no protected attributes** (no race, sex, or
ethnicity), so the disparate-impact metrics above **cannot be computed on this
data** — there are no group labels to disaggregate by. A production
implementation would obtain them via HMDA (mortgages) or BISG inference
(consumer), or a proof-of-concept could demonstrate the machinery against a
synthetic or proxy-derived group. This document is the procedure that would run
once such labels exist, alongside the preventive choices already in the model.

## Summary

| | What it does | Status here |
|---|---|---|
| No protected features | Avoids disparate *treatment* | Done |
| Geography excluded | Removes the main race proxy | Done |
| SHAP reason codes | Supports adverse-action notices | Done |
| Outcome/error analysis by group | Detects disparate *impact* | Requires protected labels |
| Four-fifths screen + significance | Flags material gaps | Requires protected labels |
| Business-necessity + LDA search | Justifies or remediates gaps | Requires protected labels |

**Two principles to carry:** disparate impact is measured on *outcomes and errors
across groups at a fixed cutoff*, not on which features the model weights; and **a
proxy can't be your justification, nor a per-group cutoff your fix.**
