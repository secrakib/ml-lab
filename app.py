import os
import joblib
import pandas as pd
import streamlit as pd_streamlit # standard alias is usually just import streamlit as st
import streamlit as st

# Set Page Configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# 1. Load the Model & Preprocessing Artifacts
@st.cache_resource
def load_assets():
    # Make sure these files are in the same directory as app.py when deploying!
    model = joblib.load("rf_housing_model.pkl")
    encoder = joblib.load("encoder.pkl")
    categorical_cols = joblib.load("categorical_cols.pkl")
    return model, encoder, categorical_cols

try:
    model, encoder, categorical_cols = load_assets()
except Exception as e:
    st.error("Error loading model artifacts. Make sure 'rf_housing_model.pkl', 'encoder.pkl', and 'categorical_cols.pkl' are in the same folder.")
    st.stop()

# 2. Application UI Header
st.title("🏠 House Price Prediction App")
st.write("Adjust the features below to predict the estimated market price of a house.")

st.markdown("---")

# 3. Create Inputs (Organized in Columns for a cleaner layout)
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=6, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=4, value=1)
    stories = st.slider("Stories/Floors", min_value=1, max_value=4, value=1)
    parking = st.slider("Parking Spaces", min_value=0, max_value=3, value=1)

with col2:
    mainroad = st.selectbox("On Main Road?", ["yes", "no"])
    guestroom = st.selectbox("Has Guestroom?", ["yes", "no"])
    basement = st.selectbox("Has Basement?", ["yes", "no"])
    hotwaterheating = st.selectbox("Has Hot Water Heating?", ["yes", "no"])
    airconditioning = st.selectbox("Has Air Conditioning?", ["yes", "no"])
    prefarea = st.selectbox("Is in Preferred Area?", ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# 4. Trigger Prediction on Button Click
st.markdown("---")
if st.button("Predict Price", type="primary", use_container_width=True):
    # Construct the input data row matching the training format
    input_data = {
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    }
    
    # Convert to DataFrame
    df_new = pd.DataFrame(input_data)
    
    # Encode categorical columns using the loaded pre-fitted encoder
    df_new[categorical_cols] = encoder.transform(df_new[categorical_cols])
    
    # Run the Inference
    prediction = model.predict(df_new)[0]
    
    # Display Result
    st.success(f"### Estimated Value: **${prediction:,.2f}**")