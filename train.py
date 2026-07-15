import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# 1. Load Dataset
df = pd.read_csv("Housing (1).csv")

# 2. Separate Features and Target
X = df.drop(columns=["price"])
y = df["price"]

# 3. Handle Categorical Features (Encoding text to numbers)
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 4. Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the Best Performing Simple Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate performance quickly
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

# 6. Save Artifacts for Inference
joblib.dump(model, "rf_housing_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")
print("Model and preprocessors saved successfully!")