
import pickle
import random

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = load_model("churn_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

# --- Defaults (used on first load) ---
GEOGRAPHIES = ["France", "Germany", "Spain"]
GENDERS = ["Male", "Female"]
YES_NO = ["Yes", "No"]

# Initialise session-state keys so widgets can read from them
defaults = {
    "credit_score": 650,
    "geography": "France",
    "gender": "Male",
    "age": 35,
    "tenure": 3,
    "balance": 50000.0,
    "num_products": 1,
    "has_cr_card": "Yes",
    "is_active": "Yes",
    "salary": 60000.0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def generate_random_customer():
    """Populate session-state with realistic random values."""
    st.session_state["credit_score"] = random.randint(350, 850)
    st.session_state["geography"] = random.choice(GEOGRAPHIES)
    st.session_state["gender"] = random.choice(GENDERS)
    st.session_state["age"] = random.randint(18, 80)
    st.session_state["tenure"] = random.randint(0, 10)
    st.session_state["balance"] = round(random.uniform(0, 250000), 2)
    st.session_state["num_products"] = random.randint(1, 4)
    st.session_state["has_cr_card"] = random.choice(YES_NO)
    st.session_state["is_active"] = random.choice(YES_NO)
    st.session_state["salary"] = round(random.uniform(10000, 200000), 2)


st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to estimate their probability of churning.")

# --- Random customer button ---
st.button("🎲 Generate Random Customer", on_click=generate_random_customer)

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 900, key="credit_score")
    geography = st.selectbox("Geography", GEOGRAPHIES, key="geography")
    gender = st.selectbox("Gender", GENDERS, key="gender")
    age = st.slider("Age", 18, 100, key="age")
    tenure = st.slider("Tenure (years with bank)", 0, 10, key="tenure")

with col2:
    balance = st.number_input("Balance", min_value=0.0, step=1000.0, key="balance")
    num_products = st.slider("Number of Products", 1, 4, key="num_products")
    has_cr_card = st.selectbox("Has Credit Card?", YES_NO, key="has_cr_card")
    is_active = st.selectbox("Is Active Member?", YES_NO, key="is_active")
    salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0, key="salary")


if st.button("Predict Churn", type="primary"):
    # Build a single-row dataframe matching the raw columns before dummy-encoding
    raw = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
        "Geography": geography,
        "Gender": gender,
    }])

    # One-hot encode the same way training data was encoded
    raw = pd.get_dummies(raw, columns=["Geography", "Gender"], drop_first=True)

    # Add any missing dummy columns (e.g. if this customer is France/Male,
    # the dummy columns for Germany/Spain/Female won't be created above)
    for col in feature_columns:
        if col not in raw.columns:
            raw[col] = 0

    # Ensure column order exactly matches training
    raw = raw[feature_columns]

    scaled = scaler.transform(raw)
    prob = float(model.predict(scaled, verbose=0)[0][0])

    st.divider()
    st.metric("Churn Probability", f"{prob:.1%}")

    if prob > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.progress(prob)

