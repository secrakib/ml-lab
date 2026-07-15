import joblib
import pandas as pd

# 1. Load the Saved Model and Artifacts
model = joblib.load("rf_housing_model.pkl")
encoder = joblib.load("encoder.pkl")
categorical_cols = joblib.load("categorical_cols.pkl")

# 2. New Unseen Data (Example input matching the dataset structure)
new_house_data = {
    "area": [7420],
    "bedrooms": [4],
    "bathrooms": [2],
    "stories": [3],
    "mainroad": ["yes"],
    "guestroom": ["no"],
    "basement": ["no"],
    "hotwaterheating": ["no"],
    "airconditioning": ["yes"],
    "parking": [2],
    "prefarea": ["yes"],
    "furnishingstatus": ["furnished"],
}

# Convert to DataFrame
df_new = pd.DataFrame(new_house_data)

# 3. Preprocess the New Data using the Trained Encoder
df_new[categorical_cols] = encoder.transform(df_new[categorical_cols])

# 4. Predict the Price
predicted_price = model.predict(df_new)

print(f"Predicted House Price: ${predicted_price[0]:,.2f}")