# app.py

import streamlit as st
import numpy as np
import joblib

# Load artifacts
@st.cache_resource
def load_artifacts():
    model  = joblib.load('diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_artifacts()

st.title("Diabetes Prediction")
st.markdown("Enter the 8 features below and click **Predict**.")

# inputs
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose      = st.number_input("Glucose (mg/dL)", 0, 300, 85)
    bp           = st.number_input("Blood Pressure (mm Hg)", 0, 200, 66)
    skin         = st.number_input("Skin Thickness (mm)", 0, 100, 29)
with col2:
    insulin      = st.number_input("Insulin (µU/mL)", 0, 900, 0)
    bmi          = st.number_input("BMI", 0.0, 70.0, 26.6, format="%.1f")
    dpf          = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.351, format="%.3f")
    age          = st.number_input("Age (years)", 0, 120, 31)

if st.button("Predict"):
    x = np.array([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    if pred == 1:
        st.error("🔴 Diabetic")
    else:
        st.success("🟢 Non-diabetic")
