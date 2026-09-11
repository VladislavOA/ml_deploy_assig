import os
import requests
import streamlit as st

# Configurable FastAPI endpoint (supports both local and Docker environments)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("Customer Churn Prediction")
st.write("Enter customer attributes below to predict whether they are likely to churn.")

# Input fields arranged in two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    OnlineBackup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )

with col2:
    DeviceProtection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    TechSupport = st.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    StreamingTV = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    StreamingMovies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )
    Contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    MonthlyCharges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=29.85, step=1.0
    )
    TotalCharges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=20000.0, value=29.85, step=10.0
    )

# Predict button
if st.button("Predict Churn", type="primary"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    try:
        response = requests.post(FASTAPI_URL, json=input_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            churn_prob = result.get("churn_probability", 0.0)

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(
                    f"⚠️ **High Churn Risk!** The model predicts this customer will **Churn** (Probability: {churn_prob:.1%})"
                )
            else:
                st.success(
                    f"✅ **Low Churn Risk!** The model predicts this customer will **Stay** (Churn Probability: {churn_prob:.1%})"
                )
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(
            f"Could not connect to model API at `{FASTAPI_URL}`. Make sure the API service is running!"
        )
