# Bangladesh Apartment Price Predictor

ML project that predicts apartment prices in Bangladesh (Taka, ৳) and ships as a Streamlit web app.

## What's inside

```
bd_apt_price/
├── data_utils.py          # cleaning + feature engineering (shared by train & app)
├── train.py               # trains LightGBM, saves model + metrics + plots
├── app.py                 # Streamlit front-end
├── requirements.txt
└── model_artifacts/       # produced by train.py
    ├── model.pkl
    ├── metadata.json
    ├── metrics.json
    ├── feature_importance.csv / .png
    └── prediction_vs_actual.png
```

## Dataset

3,865 Bangladesh apartment listings scraped from property sites. The model
uses these inputs:

- **Specs:** Bedrooms, Bathrooms, Floor_no, Floor_area (sq ft)
- **Status:** Occupancy (vacant / occupied)
- **Location:** City + Neighborhood (extracted from `Location`)

Target: `Price_in_taka` (Taka).

The cleaning pipeline parses Bangla-style prices (`'৳39,000,000'`), weird
floor numbers (`'G+7'`, `'8th'`), backfills missing Bedrooms / Floor_area
from the listing title, and imputes Bathrooms from Bedrooms.

## Run it

```bash
pip install -r requirements.txt

# 1. Train (uses the CSV at /workspace/attachments/... by default)
python train.py /workspace/attachments/a981df65__d66d341e-e29a-4874-9ec6-b7c92fcee00f.csv

# 2. Launch the Streamlit app
streamlit run app.py
```

## Model

- **Algorithm:** LightGBM (gradient-boosted trees) on `log1p(price)` — prices
  span ~৳1M to ৳500M, so log-target keeps residuals sane.
- **Validation metrics:** R² = 0.92, MAE ≈ ৳1.6M, MAPE ≈ 15%.

Strongest signals (by gain): City, Floor_area, neighborhood, Bedrooms.
