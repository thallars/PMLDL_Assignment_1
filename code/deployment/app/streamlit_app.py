import streamlit as st
import requests
import os

FASTAPI_URL = "http://fastapi:8000/predict"

st.title("Simple 4-feature classifier")
st.write("Enter 4 numeric features and press Predict")

f1 = st.number_input("Feature 1", value=0.0, format="%.6f")
f2 = st.number_input("Feature 2", value=0.0, format="%.6f")
f3 = st.number_input("Feature 3", value=0.0, format="%.6f")
f4 = st.number_input("Feature 4", value=0.0, format="%.6f")

if st.button("Predict"):
    input_data = {"f1": float(f1), "f2": float(f2), "f3": float(f3), "f4": float(f4)}
    try:
        response = requests.post(FASTAPI_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"The model predicts: {prediction}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
