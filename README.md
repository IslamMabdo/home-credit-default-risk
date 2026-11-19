# Home Credit Default Risk — Stage-4 Credit Scoring Pipeline

A clean, leak-free, production-ready tabular machine learning project predicting customer loan default probabilities (**ROC-AUC**) on the 32-feature GCI World 2026 Home Credit Default Risk dataset.

This repository intentionally stops at the **Stage-4 80/20 sweet spot**—capturing **~99% of our full 9-model competition ensemble performance** (`~0.7718` CV / `~0.774` LB vs. `0.7729` CV / `0.7751` LB) with **zero over-engineering**, clear domain-justified features, full test coverage, and standalone inference artifacts.

---

## Why Stage 4? (Controlled 5-Fold / 10-Fold Ablation Progression)

| Stage | Description | OOF ROC-AUC | Incremental Gain | % of Max Gain |
| :--- | :--- | :---: | :---: | :---: |
| 0 | Baseline 5-feature Random Forest | `~0.668` | — | — |
| 1 | All 32 raw features + default LightGBM | `0.74985` | `+0.0819` | 78% |
| 2 | Core financial/temporal ratios + anomaly cleaning (`365243`, car age) | `0.75911` | `+0.0093` | 87% |
| 3 | Target-free OOF `EXT_SOURCE_1/2/3` imputation + peer group stats | `0.76670` | `+0.0076` | 94% |
| **4 (This Repo)** | **Discrete loan-product nested-OOF TE + shallow regularized `XGBoost` (`0.77060`) / `LightGBM` (`0.77048`) / 50-50 blend (141 features, 3.9 min on RTX 4050)** | **`0.77075`** | **`+0.0041`** | **~98%** |
| 5 *(Competition-only)* | Full 9-model dual-view ensemble (2 GPU NNs + Zero-TE CatBoost + Stage-2 distillation + 10-fold) | `0.77288` | `+0.0011` | 100% |

---

## Key Domain & Data Discoveries

1. **Sentinel Anomalies:** `DAYS_EMPLOYED == 365243` (~18% of rows, pensioners/unemployed at `5.42%` default vs `8.66%` active workers) and `OWN_CAR_AGE >= 60` (`64`/`65` sentinel values) are separated into explicit flags + cleaned numeric columns.
2. **`EXT_SOURCE_3 × Credit Bureau` Missingness Split:** Missing `EXT_SOURCE_3` splits into two completely different populations—clients who still have Credit Bureau enquiry records (`7.74%` default, matching non-missing `7.71%`) vs. thin-file clients missing both (`10.38%` default).
3. **Target-Free `EXT_SOURCE` Reconstruction:** `EXT_SOURCE_1` (69.5% missing) is predictable at Pearson $r \approx 0.70$ from non-target features alone; target-free OOF regression provides both filled scores and credit-surprise residuals (`actual - expected`).
4. **Discrete Loan-Product Risk Tiers:** Adjacent loan values carry up to 10× different default rates (e.g., `AMT_GOODS_PRICE = 450k` at `13.99%` vs. `454.5k` at `5.86%`; `CREDIT/ANNUITY = 10.1` at `1.98%` vs. `12.6` at `19.62%`). Finer histogram bins (`max_bin=512`) + nested-OOF smoothed target encodings recover this signal without leakage.

---

## Repository Structure

```text
home-credit-default-risk/
├── configs/
│   └── config.yaml                     # Centralized hyperparameters, paths & feature settings
├── notebooks/
│   └── exploration_and_modeling.ipynb  # Story-driven EDA, ablation, diagnostics & slice analysis
├── src/credit_risk/
│   ├── config.py                       # Typed dataclass YAML configuration loader
│   ├── features.py                     # Schema validation, stateless domain FE, ExtSourceImputer & FoldEncoder
│   ├── model.py                        # Leak-free Stratified K-Fold trainer & standalone CreditRiskBundle
│   ├── evaluate.py                     # ROC-AUC, PR-AUC, Brier, Bootstrap 95% CI & slice analysis
│   └── pipeline.py                     # Production CLI (train / predict / evaluate)
├── tests/
│   └── test_pipeline.py                # Unit & end-to-end tests (schema, determinism, zero-leakage, serving)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[viz,dev]"
```

Place `train.csv` and `test.csv` inside `data/`.

### 2. Train & Save Standalone Production Bundle
```bash
credit-risk train --config configs/config.yaml
# Optional: use GPU or switch model type (xgboost | lightgbm | blend)
credit-risk train --config configs/config.yaml --model-type blend --gpu
```
Saves `artifacts/model_bundle.joblib`, `artifacts/metrics.json`, `artifacts/oof_predictions.csv`, and `artifacts/slice_metrics.csv`.

### 3. Batch Inference on Unseen Applications (No `train.csv` Needed)
```bash
credit-risk predict --config configs/config.yaml --input data/test.csv --output outputs/submission.csv
```

### 4. Inspect Metrics & Run Tests
```bash
credit-risk evaluate --config configs/config.yaml
pytest
```

---

## Production Readiness Highlights
* **Zero Train/Serve Skew:** Both the notebook and CLI share the exact same `credit_risk.features` module.
* **Standalone Inference Artifact:** `CreditRiskBundle` serializes frozen categorical vocabularies, fitted `EXT_SOURCE` imputers, and per-fold `(FoldEncoder, model)` lookup tables—scoring new loans without ever reading `train.csv`.
* **Leakage Prevention:** All target encodings use nested inner-5-fold smoothing strictly inside each training fold; `EXT_SOURCE` imputation never uses `TARGET`.
* **Input Contract Enforcement:** `validate_schema()` checks required columns, unique IDs, and binary target integrity before execution.
