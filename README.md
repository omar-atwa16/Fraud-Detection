# Fraud Detection

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Quality Check](#data-quality-check)
- [Main Data Issues](#main-data-issues)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Online payment fraud is rare but costly. This project builds a pipeline that:

1. Loads and inspects a transaction dataset
2. Runs an automated **data-quality check** (missing values, types, duplicates, cardinality)
3. Cleans and prepares features
4. Trains and evaluates models to flag fraudulent transactions

The main challenges are **many missing values**, **hundreds of anonymized features**, and **heavy class imbalance** (very few fraud cases compared with legitimate ones).

---

## Dataset

The data is a table of card transactions. Column groups:

| Column(s) | Description |
|---|---|
| `TransactionDT` | Time delta from a reference date (not a real timestamp) |
| `TransactionAmt` | Transaction amount |
| `ProductCD` | Product code for the transaction |
| `card1` – `card6` | Payment card information (type, category, issuer, etc.) |
| `addr1`, `addr2` | Billing address information |
| `dist1`, `dist2` | Distance-related features |
| `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domains |
| `C1` – `C14` | Counting features (e.g. how many addresses are linked to the card) |
| `D1` – `D15` | Timedelta features (e.g. days since a previous transaction) |
| `M1` – `M9` | Match flags (e.g. name on card matches address) |
| `V1` – `Vxxx` | Engineered, anonymized features |

- **Target:** `isFraud` (1 = fraud, 0 = legitimate) — *update if your target column is named differently*
- **Identifier:** `TransactionID` — *update if different*
- **Source:** *add dataset source / link here*
- **Size:** *add number of rows and columns here*


---

### Add the data

Download the dataset from *[add source link]* and place the files in `data/raw/`.

---

## Data Quality Check

The data-quality script inspects the dataset and writes a summary report.

Output is saved to `Data Quality Output.txt`. It covers:

- Number of rows and columns
- Data types per column
- Missing values per column (count and percentage)
- Duplicate rows
- Unique value counts (cardinality)
- Class balance of the target

---

## Main Data Issues

Found during the data-quality check:

- **Missing values:** many columns, especially the `D`, `M` and `V` groups, have a high share of missing values
- **Class imbalance:** fraud is a small fraction of all transactions, so accuracy alone is a misleading metric
- **Anonymized features:** `V` columns have no description, which limits interpretation
- **High-cardinality columns:** `card` and email-domain columns need careful encoding
- **Non-real timestamp:** `TransactionDT` is a relative time delta, so time features need care

---

## Tech Stack

- **Language:** Python
- **Data:** pandas, NumPy
- **Modeling:** scikit-learn *(add XGBoost / LightGBM if used)*
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebook

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.