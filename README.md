# Predictive Modelling of Loan Defaults

This repository contains an end-to-end machine learning pipeline to predict U.S. SBA loan defaults using ~900,000 records (1962–2014).
It demonstrates practical ML for credit risk, with emphasis on minimising undetected defaults, calibrated probabilities, and fairness/regulatory alignment.

## Highlights
- **Dataset:** ~900k historical SBA loans (1962–2014). (Raw data not included; see “Data”.)
- **Pipeline:** data cleaning → feature engineering → 70/30 train/test split → class-imbalance experiments → cross-validation → model selection.
- **Models evaluated:** Logistic Regression, K-Nearest Neighbours, Random Forest, Support Vector Machines.
- **Selected model:** **Calibrated Random Forest** evaluated at a **0.20** decision threshold.
  - **AUC ≈ 0.956**
  - **Recall (TPR) ≈ 89–90%**
  - **FPR ≈ 11%**
  - **Accuracy ≈ 88.9%**
  - **Precision ≈ 63.1%**
  - **F1 ≈ 73.9%**
- **Threshold selection:** 0.20 chosen to materially reduce false positives while maintaining high recall; overall accuracy and F/G-scores improved versus a 0.10 threshold.
- **Explainability:** permutation importance and SHAP to support auditability and model governance.
- **Regulatory framing:** modelling choices aligned with EBA/BIS/EBF guidance on calibrated, transparent, and fair credit-risk models.

> Business aim: Minimise undetected defaulters (FN) without over-rejecting safe applicants (FP), using calibrated and explainable predictions for decisioning.

## Repository Structure
```

.
├── notebooks/                 # Jupyter notebooks (add your .ipynb here)
├── src/                       # Optional Python modules/utilities
├── figures/                   # Export plots (ROC, PR, SHAP)
├── data/                      # Do not commit large/raw data (ignored by .gitignore)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignore data/env/temporary files
├── LICENSE                    # MIT
└── README.md                  # Project overview

````

## Quickstart
1. Clone or download this repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ````

3. Launch Jupyter and open the notebook(s) in `notebooks/`:

   ```bash
   jupyter notebook
   ```

## Reproducing Results (outline)

* Perform data cleaning and feature engineering; guard against leakage.
* Train and evaluate Logistic Regression, KNN, Random Forest, and SVM.
* Calibrate probabilities (Platt scaling) and evaluate at a 0.20 decision threshold.
* Report metrics: TPR/FPR, Precision/Recall, F1, AUC, and confusion matrices.
* Use SHAP for interpretability; save plots to `figures/`.

## Key Results

* **Calibrated Random Forest (threshold 0.20):** AUC ≈ 0.956, TPR ≈ 89–90%, FPR ≈ 11%, Accuracy ≈ 88.9%, Precision ≈ 63.1%, F1 ≈ 73.9%.
* Compared with a 0.10 threshold, the 0.20 threshold substantially reduces false positives while preserving high recall and improving overall summary scores.

## Ethics and Fairness

* Sensitive attributes (e.g., Minority) excluded to avoid discriminatory bias and to align with regulatory expectations.
* SHAP-based explanations used to support auditability and transparent decision support.

## Data

* Raw dataset not included due to size/licensing. Small public sample (1,000 rows) for demonstration in `data/`.

## Notebooks

* Main analysis: `notebooks/loan_default_prediction.ipynb`

## Author

**David Crowe** — BSc Economics & Finance (UCD) | MSc Quantitative Finance (UCD Smurfit)
