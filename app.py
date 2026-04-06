import streamlit as st
import numpy as np
import pandas as pd
import pickle


feature_names = pickle.load(open("feature_names.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer details:")



tenure = st.number_input("Tenure", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", [
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
])

partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])



partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
online_security = 1 if online_security == "Yes" else 0



contract_One_year = 1 if contract == "One year" else 0
contract_Two_year = 1 if contract == "Two year" else 0

internet_Fiber = 1 if internet == "Fiber optic" else 0
internet_No = 1 if internet == "No" else 0

payment_Electronic = 1 if payment == "Electronic check" else 0
payment_Mailed = 1 if payment == "Mailed check" else 0
payment_Bank = 1 if payment == "Bank transfer (automatic)" else 0
payment_Credit = 1 if payment == "Credit card (automatic)" else 0

input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Partner": partner,
    "Dependents": dependents,
    "OnlineSecurity": online_security,
    "Contract_One year": contract_One_year,
    "Contract_Two year": contract_Two_year,
    "InternetService_Fiber optic": internet_Fiber,
    "InternetService_No": internet_No,
    "PaymentMethod_Electronic check": payment_Electronic,
    "PaymentMethod_Mailed check": payment_Mailed,
    "PaymentMethod_Bank transfer (automatic)": payment_Bank,
    "PaymentMethod_Credit card (automatic)": payment_Credit
}

input_df = pd.DataFrame([input_dict])

# Align with training columns
input_df = input_df.reindex(columns=feature_names, fill_value=0)

input_scaled = scaler.transform(input_df)
if st.button("Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob > 0.7 else 0

    if prediction == 1:
        st.error(f"Customer will CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"Customer will NOT churn (Probability: {prob:.2f})")