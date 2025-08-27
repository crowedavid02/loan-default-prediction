# Predictive Modelling of Loan Defaults

This repository contains an end-to-end machine learning pipeline to predict **U.S. SBA loan defaults** using ~900,000 records (1962–2014).  
It demonstrates practical ML for **credit risk**, with emphasis on *minimising false negatives*, **calibration**, and **fairness/regulatory alignment**.

## Highlights
- **Dataset:** ~900k historical SBA loans (1962–2014). *(Raw data not included; see “Data” section.)*
- **Pipeline:** data cleaning → feature engineering → train/test split → class imbalance handling → cross-validation → model selection.
- **Models:** Logistic Regression, KNN, Random Forest, SVM.
- **Best model:** **Calibrated Random Forest**, threshold = **0.20** → **AUC ≈ 0.96**, **~90% sensitivity (TPR)**, **~40% reduction in false negatives** (vs. 0.10 threshold trade-offs).
- **Explainability:** SHAP + permutation importance for transparency.
- **Compliance:** results framed with **EBA/BIS** guidance on explainable/ethical AI in credit scoring.

> Business aim: Reduce undetected defaulters (FN) while keeping FPR commercially viable and maintaining fairness.

## Repo Structure
```
.
├── notebooks/                 # Jupyter Notebooks (add your .ipynb here)
├── src/                       # Optional: python modules/utilities
├── figures/                   # Export plots here (ROC, PR, SHAP)
├── data/                      # DO NOT COMMIT large/raw data (see .gitignore)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignore large files & envs
├── LICENSE                    # MIT (for your code)
└── README.md                  # You are here
```

## Quickstart
1. **Clone** or **download ZIP** of this repo.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Add your notebook(s) into `notebooks/` and run in Jupyter:
   ```bash
   python -m jupyter notebook
   ```

## Reproducing Results (outline)
- Load preprocessed features (or perform cleaning/feature engineering).
- Train and evaluate Logistic Regression, KNN, Random Forest, SVM.
- Calibrate probabilities (Platt scaling) and evaluate at **0.20** decision threshold.
- Report metrics: **TPR/FPR**, **F1**, **AUC**, and confusion matrices.
- Use **SHAP** for explainability; save plots to `figures/` and reference them below.

## Key Results (from report)
- Calibrated Random Forest: **AUC ≈ 0.96**, **TPR ~89–90% at 0.20 threshold**, improved precision vs. 0.10 while sharply reducing FPR.
- Threshold selection motivated by **cost asymmetry** of FN vs. FP in lending and **regulatory expectations** for calibrated/transparent models.

## Ethical/Regulatory Notes
- Sensitive attributes (e.g., *Minority*) were **excluded** to avoid discrimination (EBA/EBF guidance).  
- Model choices prioritised explainability and probability calibration to support audited lending decisions.

## Data
- **Raw dataset not included** due to size/licensing. Provide instructions or links for users to download the SBA dataset separately.
- Optionally include a small **sample (e.g., 1,000 rows)** for demonstration in `data/`.

## Useful Links
- Add links to project report/slides or hosted visuals if available.

## Author
**David Crowe** — BSc Economics & Finance (UCD) | MSc Quantitative Finance (UCD Smurfit)  
Feel free to connect on LinkedIn.
