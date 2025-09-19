import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# Streamlit app UI
st.title("Wine Quality Classifier")

st.write("Enter the physicochemical properties of the wine to predict its quality.")

# Input fields for wine data
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)

# Make prediction when the button is clicked
if st.button("Predict"):
    # Prepare the data for the API request
    input_data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    # Send a request to the FastAPI prediction endpoint
    try:
        response = requests.post(FASTAPI_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"The model predicts wine quality: {prediction}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
