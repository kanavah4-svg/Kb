
# ATELIER 8 – Data Analytics Dashboard

This repository contains a simple Streamlit dashboard for the **ATELIER 8** Data Analytics Individual PBL project.

The app lets you:
- Upload your survey CSV data
- Explore basic statistics and visualizations
- Build K-Means clusters for customer segments
- Train a classification model to predict adoption intent
- Run a regression model to understand willingness to pay (WTP)

## Files in this repo

- `app.py` – Main Streamlit app
- `requirements.txt` – Python dependencies for Streamlit Cloud / GitHub deployment
- `sample_data.csv` – Tiny sample dataset so the app can run even without your file
- `README.md` – This file

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected data structure

Your CSV should ideally include:

- **Numeric columns** such as:
  - `age`
  - `income_level`
  - `num_luxury_items`
  - `wtp_restoration` (willingness to pay for restoration)
  - `wtp_authentication` (willingness to pay for authentication)
  - `sustainability_score` (1–5 scale, for example)

- **Categorical columns** such as:
  - `adoption_intent` (e.g., "Very likely", "Likely", etc.)
  - `gender`
  - `brand_preference`

The app automatically detects numeric vs categorical columns and lets you choose which ones to use for clustering, classification, and regression.

If your column names are different, you can either:
1. Rename them in your CSV to match the above, **or**
2. Just use whatever numeric / categorical columns your dataset already has.

## Deployment (GitHub + Streamlit Cloud)

1. Create a new GitHub repository.
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `sample_data.csv`
   - `README.md`
3. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Connect your GitHub, select the repo and `app.py` as the main file.
5. Deploy – your dashboard URL will be generated automatically.

You can then upload your full UAE survey dataset from the browser and use all features of the dashboard.
