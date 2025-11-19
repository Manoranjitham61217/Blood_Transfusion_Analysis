import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("AdaBoost_model.pkl") 
scaler = joblib.load("Scaled_transfusion.pkl")

st.set_page_config(page_title="Blood Transfusion Predictor", layout="wide")

st.title("🩸 Blood Transfusion Donation Prediction (AdaBoost)")
st.write("Predict whether a person will donate blood again.")

col1, col2 = st.columns(2)
with col1:
    Recency = st.number_input("Recency (months since last donation)", 0, 100, 5)
    Frequency = st.number_input("Frequency (total donations)", 0, 100, 2)

with col2:
    Monetary = st.number_input("Monetary (total c.c. blood donated)", 0, 5000, 500)
    Time = st.number_input("Time (months since first donation)", 0, 200, 20)

input_data = np.array([[Recency, Frequency, Monetary, Time]])
scaled_data = scaler.transform(input_data)
if st.button("Predict"):
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.success(f"💉 Likely to donate again! (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Not likely to donate again. (Probability: {prob:.2f})")
