# Lending Club — Credit-Risk Default Model

Predict, using **only information known at loan application**, whether a
Lending Club loan will default (charge off) rather than be repaid — then turn
that model into an approve/decline decision with an **expected-loss / P&L**
justification.

This is a credit-risk portfolio project: model → decision → dollars. The
central discipline is **avoiding data leakage** (never train on anything that
is only known after a loan is originated).

## Status

- [x] **Session 1 — setup & data triage** (in progress)
  - [x] Project scaffold, environment, download + load pipeline
  - [ ] Download dataset (needs Kaggle token — see below)
  - [ ] Load into SQLite, confirm shape
  - [ ] Keep / drop / leakage column triage (fill in the table below)
- [ ] Session 2 — target definition, train/test split, logistic baseline
- [ ] Session 3 — model bake-off (LogReg → RandomForest → XGBoost/LightGBM) + expected-loss threshold
- [ ] Session 4 — SHAP interpretability
- [ ] Session 5 — Streamlit app + written recommendation

## Dataset

Kaggle **`wordsforthewise/lending-club`** — the most complete current mirror.
We model `accepted_2007_to_2018Q4.csv.gz` (~2.2M funded loans, 151 columns).
Rejected applications are out of scope for now.

## Setup

```bash
# 1. Create the environment (Python 3.14, venv already scaffolded)
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Give Kaggle an API token (one-time). Either:
./venv/bin/kaggle auth login                 # OAuth, browser prompt
# ...or create a token at https://www.kaggle.com/settings -> API -> Create New Token
#    then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 3. Download + load (or just run the notebook, which does both)
./venv/bin/python src/download_data.py        # -> data/raw/accepted_2007_to_2018Q4.csv.gz
./venv/bin/python src/load_to_sqlite.py       # -> data/lending_club.db (table `loans`)

# 4. Explore
./venv/bin/jupyter lab notebooks/01_load_and_triage.ipynb
```

The venv is registered as a Jupyter kernel named **"Python (Lending Club)"** —
select it in the notebook. `requirements-lock.txt` pins exact versions.

> **macOS note:** `xgboost`/`lightgbm` need the OpenMP runtime. If they fail to
> import with a `libomp.dylib` error, run `brew install libomp`.

## Layout

```
├── data/
│   ├── raw/                  # downloaded CSV(s) — gitignored
│   └── lending_club.db       # SQLite, table `loans` — gitignored
├── notebooks/
│   └── 01_load_and_triage.ipynb
├── src/
│   ├── download_data.py      # Kaggle download
│   ├── load_to_sqlite.py     # chunked CSV -> SQLite
│   └── lc_data_dictionary.py # column descriptions + leakage flags
├── reports/
│   └── data_dictionary.*     # generated: annotated triage worksheet
├── requirements.txt / requirements-lock.txt
```

## The leakage rule

For every column ask: **would a lender know this the day the application
arrives?** If not, exclude it. Flags used in the data dictionary:

| Flag | Meaning | Action |
|------|---------|--------|
| `TARGET` | The outcome (`loan_status`) | Build the label from it, don't use as feature |
| `SAFE` | Known at application time | Use as a predictor |
| `LC_MODEL` | Known at application, but is LC's own risk output (`grade`, `sub_grade`, `int_rate`) | Use deliberately — see note |
| `LEAKAGE` | Only known after origination (`recoveries`, `total_pymnt`, `out_prncp`, ...) | **Exclude** |
| `REVIEW` | Judgment call | Decide during triage |

**On `grade`/`int_rate`:** these are legitimate (set before funding) but they
*are* LC's own risk model. Plan: train one model with them and one without, to
measure how much of your signal is just re-learning LC's grade.

## Column triage

_Generated worksheet: `reports/data_dictionary.csv`. Record final decisions and
reasoning here once the data is loaded._

| Decision | Columns | Reasoning |
|----------|---------|-----------|
| **Drop (leakage)** | _tbd_ | Post-origination |
| **Keep (features)** | _tbd_ | Known at application |
| **Use deliberately** | `grade`, `sub_grade`, `int_rate` | LC's own risk output |
| **Target** | `loan_status` | Charged Off/Default = 1, Fully Paid = 0 |
