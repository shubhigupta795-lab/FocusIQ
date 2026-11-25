# FocusIQ 4.0 — Digital Wellness & Productivity Intelligence

Complete local semester-five data science project. This release consolidates the earlier fixes, uses a white dashboard, removes the challenge and 24-hour planner, and retrains the classifiers for three classes. All source is readable and formatted. No additional dataset or cloud service is required.

## Run in VS Code (Windows)

Extract this archive into a **new folder**, open `FocusIQ_Final` in VS Code, and run the following from its terminal. Use 64-bit Python 3.13 (build tested with 3.13.5). If a previous server is running, stop it with Ctrl+C first.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and refresh with Ctrl+F5. Do not open index.html directly. Plotly is served locally by FastAPI. Model files are already included; no training is required on startup. To retrain, use Model Lab or `python train.py` with the server stopped, using the same environment. API documentation is at `/docs`.

## Dataset and limitations

The original uploaded `student_productivity_distraction_dataset_20000.csv` is included unchanged: 20,000 unique IDs and 18 columns. Source previously identified as Student Productivity & Digital Distraction Dataset on Kaggle (sehaj1104); data-collection provenance is unverified. No additional synthetic training records. The CSV has no dates, notifications, phone checks, nighttime use or intervention groups. 15,458 records report summed social/YouTube/gaming hours above phone time, so individual app times are treated as separate reported measures. No causal claims. Optional personal inputs affect wellness only, not the ML model. The score is a heuristic, not medically validated.

## Eight pages

- Overview: white dashboard, filters, live what-if score and predictions, phone buddy text, actual histogram, scatter/regression line, box plot, normalized habit heatmap, five findings and held-out SHAP importance.
- My wellness: full profile, percentile, three-class probabilities, regression estimate, SHAP, PDF and scenario comparison.
- Data Studio: audit, descriptive statistics, IQR flags, histogram, box plot, age-coloured scatter plots and six-panel pairplot, Pearson heatmap. Blue circles = ages 17–20, amber diamonds = 21–25, green squares = 26–29. Plot samples are deterministic display subsets, not training subsets.
- Research Lab: dataset-based Z, Welch t, Chi-square, ANOVA, conditional probability and bootstrap CI. Pilot experiment removed.
- Probability Lab: Normal, Binomial, Poisson and Uniform simulation, Type I/II errors, power and parameter controls.
- Behavioural Clusters: same 2,000 sampled students, standardized ten numeric features, K-Means and Ward hierarchical clustering, four groups. Profile names are derived from the two largest absolute standardized deviations in six habit means relative to the sample. Cards, population bars, grouped habit bars, profile table and silhouette comparison. PCA retained as variance calculation; no scatter plot or dendrogram.
- Model Lab: 11 classifier configurations, grouped three-fold CV, random oversampling, SMOTENC, Grid Search, Randomized Search, ensemble models, XGBoost, macro metrics, model selector, three-class confusion matrix and three one-vs-rest ROC curves for each classifier. The selector is for evaluation; deployed model is selected by CV macro F1. Linear regression still estimates the numerical score separately.
- Monitoring: saved assessment dropdown, actual score entry, class, prediction error, version-specific MAE, input drift and daily MAE trend (three-observed-day rolling mean once available). Uses real saved outcomes, not invented timestamps. Challenge tracking and pilot APIs removed.

## Classification

Low = score <40; Normal = 40–<70; High = 70–100. These are project-defined thresholds, not clinical labels. Actual dataset class counts: 5,536 / 12,143 / 2,321. Same participant-disjoint split: 15,600 training and 4,400 held out. All preprocessing and resampling occur inside training folds. Macro precision, recall and F1 give each class equal weight. ROC AUC is macro one-vs-rest; ROC curves use predicted probabilities on the held-out set, not class labels. Confusion matrix rows are actual and columns predicted. SHAP explains the probability of the Low class only, with categorical inputs held fixed.

## Main code to study

| File | Purpose |
|---|---|
| data_engine.py | Pandas ingestion, audit, feature engineering, class labels, heuristic wellness |
| analytics.py | Statistics, tests, bootstrap, probability simulation, clustering, silhouette and PCA |
| ml.py | Pipelines, resampling, training, cross-validation, tuning, multiclass metrics, ROC, regression and SHAP |
| app.py | FastAPI, personal predictions, monitoring, rolling daily errors, report |
| train.py | Training entry point |
| static/app.js | Navigation, charts, cluster and model selectors, monitoring |
| static/dashboard.js | Overview simulator and analytics |
| static/index.html / style.css | Eight-page light frontend |
| tests/test_app.py | Automated API and data checks |

## Course coverage changes

All seven modules remain represented. The challenge's dated behavioural forecast and paired intervention test were intentionally removed at your request. The remaining Welch t-test is a valid independent-samples t-test; a simple time-series moving average is available in monitoring only after actual dated predictions exist. The dataset does not support longitudinal behavioural forecasting. SHAP is implemented; LIME is not separately implemented. AutoML is basic automatic candidate selection, not an external AutoML package. This is a local academic prototype, not a production-hardened multi-user service.

## Validation

Run `.\.venv\Scripts\python.exe -m pytest -q` (Windows) or `python -m pytest -q` in the project environment. The six API tests cover original data, all multiclass metrics and ROC endpoints, both clustering methods, statistics/probability, personal prediction, SHAP, PDF, actual outcomes and removal of obsolete endpoints. Both JS files passed Node syntax checks (Node is not required for runtime). Core packages were tested together in Linux; Windows installation and live network launch are not independently verified. Use a fresh virtual environment to avoid unrelated global-package conflicts. The SQLite database is created at startup; no personal/test records are shipped.
