import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("LR_model.pkl")

st.title("🚗 Ford Car Price Prediction")

year = st.number_input("Year", 1996, 2023, 2018)
mileage = st.number_input("Mileage", 0, 300000, 30000)
tax = st.number_input("Tax", 0, 600, 150)
mpg = st.number_input("MPG", 0.0, 100.0, 50.0)
engineSize = st.number_input("Engine Size", 0.8, 5.0, 1.5)

model_name = st.selectbox(
    "Model",
    ["Fiesta","Focus","Kuga","EcoSport","Mondeo","Ka+","Puma","Mustang","S-MAX","C-MAX"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual","Automatic","Semi-Auto"]
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol","Diesel","Hybrid","Electric","Other"]
)

if st.button("Predict Price"):

    columns = joblib.load("columns.pkl")
    scaler = joblib.load("scaler.pkl")

    input_data = pd.DataFrame([[0] * len(columns)], columns=columns)

    input_data["year"] = year
    input_data["mileage"] = mileage
    input_data["tax"] = tax
    input_data["mpg"] = mpg
    input_data["engineSize"] = engineSize

    model_col = f"model_{model_name}"
    if model_col in input_data.columns:
        input_data[model_col] = 1

    trans_col = f"transmission_{transmission}"
    if trans_col in input_data.columns:
        input_data[trans_col] = 1

    fuel_col = f"fuelType_{fuelType}"
    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1

    input_data = pd.DataFrame(
        scaler.transform(input_data),
        columns=columns
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")