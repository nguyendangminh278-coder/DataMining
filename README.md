# Rossmann Store Sales — Data Mining Team 2

This repository consolidates the team's **Rossmann Store Sales** Data Mining code from the [DM Team 2 planning spreadsheet](https://docs.google.com/spreadsheets/d/13U0yae5CTqzcZMZgK_qKiCnD-nrIz8yUb8S04xI7j-s/edit) and its linked Drive/Colab.

**Main research questions:** (1) What is the adjusted relationship between Promo2 activity and Sales? (2) How accurately can daily store-level Sales be forecast six weeks in advance? Treat association and causality as different claims.

## Code by phase

| Phase | Notebook | Code cells |
| --- | --- | ---: |
| P05 — Basic Data Cleaning | [01_P05_Clean_Data.ipynb](notebooks/01_P05_Clean_Data.ipynb) | 6 |
| U02 — Exploratory Data Analysis | [02_U02_EDA_RootCause.ipynb](notebooks/02_U02_EDA_RootCause.ipynb) | 37 |
| P07–P14, F01, M01, E01, D01 — Promo2 | [03_F01_Promo2_Analysis.ipynb](notebooks/03_F01_Promo2_Analysis.ipynb) | 88 |
| F02–E03 — Forecasting & xAI | [04_F02_Forecasting_xAI.ipynb](notebooks/04_F02_Forecasting_xAI.ipynb) | 66 |

**Colab you supplied:** [P07 – P14 – F01 – D01 – E01 – M01](https://colab.research.google.com/drive/19K7REYOXk6DfUlXBEf50IWaSfo_a5GK6?hl=vi). This is represented by notebook 03 above; it is not omitted. Original cell sources and order are preserved. Saved execution outputs were removed from GitHub-ready notebook copies.

- `notebooks/`: Jupyter notebook versions for review/running.
- `scripts/`: Code cells exported as Python in original order for searching/editing. **These are not standalone pipelines.**
- `archive/previous_versions/`: Earlier U02 version.
- `docs/`: source links, phase map, consolidated code reference.
- `data/raw/`: keep Kaggle CSV locally (gitignored).
- `tools/`: helper scripts to validate notebook structure and prepare local data paths.

## Getting started

1. Install Python 3.10+ and `pip install -r requirements.txt`.
2. Download original [Rossmann Kaggle data](https://www.kaggle.com/competitions/rossmann-store-sales/data) (`train.csv`, `test.csv`, `store.csv`, `sample_submission.csv`) to `data/raw/`.
3. Run `python tools/prepare_data_layout.py` to create common local input aliases.
4. Run `python tools/check_notebooks.py` and open the `notebooks/` files with Jupyter or Colab.
5. Some notebooks require intermediate clean/P09/P14 datasets. Get them from the team's Drive or run the corresponding preparation steps; update their original Colab-specific file paths as necessary.

**Reproducibility limits:** This is a preservation and organization of existing team work. We have not rerun model fitting, evaluated results, or proven that each notebook is executable end-to-end with only the four raw CSVs. Do not infer causal effects from predictive importance or SHAP.

## Sharing and safety

This public repository intentionally excludes large raw/processed datasets, credentials, private documents and model artifacts. Refer to the original source links and observe their terms of use. A repository license should only be chosen after team ownership is agreed.
