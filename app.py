import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Load model and columns
model = joblib.load("random_forest_model.pk1")
model_columns = joblib.load("model_columns.pk1")

st.title("Car Price Prediction")

# User Inputs
year = st.number_input("Year", min_value=1990, max_value=2026, value=2018)

km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

fuel = st.selectbox(
    "Fuel Type",
    ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner Type",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=20.0)

engine = st.number_input("Engine (CC)", min_value=0, value=1200)

max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=80.0)

seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

name = st.text_input("Car Name", "Maruti Swift Dzire")

# Feature Engineering
current_year = datetime.now().year
car_age = current_year - year

# Create input dataframe
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats],
    'car_age': [car_age],
    'name': [name]
})

# Apply same preprocessing as notebook
input_data = pd.get_dummies(input_data)

# Match training columns
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Price"):

    prediction = np.expm1( model.predict(input_data)[0])

    st.success(f"Estimated Car Price: ₹ {prediction:,.2f}")