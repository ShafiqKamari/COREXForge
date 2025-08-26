# COREXForge - Contribution via Removal & Explained Variability for MCDM Weighting Method

This Streamlit app implements the COREX weighting method and presents the workflow in 9 steps, aligned with your document.
It supports auto-selecting a column named Alternative as the row identifier and can export all steps plus the final summary to a single Excel file.

## Features
- Upload CSV or Excel, or use a built-in sample dataset.
- Assign criterion types: Benefit, Cost, Target (with target values).
- Choose the blend parameter alpha between 0 and 1.
- View intermediate outputs step by step and download each as CSV.
- One-click Excel export containing all 9 steps plus the Summary in separate sheets.

## Steps shown in the app
1. The Decision Matrix
2. Normalization of the Decision Matrix
3. The Overall Performance Score
4. The Performance Score under Criterion Removal
5. Removal Impact Score
6. The Standard Deviation of Each Criterion
7. The Sum of Absolute Correlations for Each Criterion
8. Explained Variability Score
9. COREX Weight Scores

## Install
```bash
pip install -r requirements.txt
```

## Run locally
```bash
streamlit run app_corexforge_steps9_excel.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Set the entry point to `app_corexforge_steps9_excel.py` and include `requirements.txt`.
4. Deploy.

## Suggested layout
```
.
├── app_corexforge_steps9_excel.py   # COREXForge app with Excel export (9 steps)
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```
