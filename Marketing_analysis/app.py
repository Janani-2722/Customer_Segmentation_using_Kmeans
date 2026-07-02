import streamlit as st
import numpy as np
import pandas as pd

# Load models
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open(BASE_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Cluster descriptions
cluster_names = {
    0: ("Low Spenders, Low Balance", "Customers with minimal activity and low balance. Rarely make purchases."),
    1: ("High Cash Advance Users", "Customers who frequently withdraw cash. High risk profile."),
    2: ("Installment Buyers", "Customers who prefer buying in installments. Steady and reliable."),
    3: ("Big One-Off Purchasers", "Customers who make large one-time purchases occasionally."),
    4: ("Inactive / Dormant Users", "Customers with very little to no activity on their account."),
    5: ("Full Payment, High Credit", "Customers who pay in full regularly. Most financially healthy segment.")
}

# Page config
st.set_page_config(page_title="Customer Segment Predictor", page_icon="🎯")
st.title("🎯 Customer Segment Predictor")
st.write("Enter customer credit card details to find their segment.")

# Input form
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    BALANCE = st.number_input("Balance", min_value=0.0, value=1000.0)
    PURCHASES = st.number_input("Purchases", min_value=0.0, value=500.0)
    ONEOFF_PURCHASES = st.number_input("One-Off Purchases", min_value=0.0, value=200.0)
    INSTALLMENTS_PURCHASES = st.number_input("Installment Purchases", min_value=0.0, value=300.0)
    CASH_ADVANCE = st.number_input("Cash Advance", min_value=0.0, value=0.0)
    CREDIT_LIMIT = st.number_input("Credit Limit", min_value=0.0, value=5000.0)

with col2:
    PAYMENTS = st.number_input("Payments", min_value=0.0, value=1000.0)
    MINIMUM_PAYMENTS = st.number_input("Minimum Payments", min_value=0.0, value=200.0)
    BALANCE_FREQUENCY = st.slider("Balance Frequency", 0.0, 1.0, 0.5)
    PURCHASES_FREQUENCY = st.slider("Purchases Frequency", 0.0, 1.0, 0.5)
    ONEOFF_PURCHASES_FREQUENCY = st.slider("One-Off Purchases Frequency", 0.0, 1.0, 0.2)
    PURCHASES_INSTALLMENTS_FREQUENCY = st.slider("Installments Frequency", 0.0, 1.0, 0.3)

col3, col4 = st.columns(2)
with col3:
    CASH_ADVANCE_FREQUENCY = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0)
    CASH_ADVANCE_TRX = st.number_input("Cash Advance Transactions", min_value=0, value=0)
with col4:
    PURCHASES_TRX = st.number_input("Purchase Transactions", min_value=0, value=5)
    PRC_FULL_PAYMENT = st.slider("% Full Payment", 0.0, 1.0, 0.5)
    TENURE = st.number_input("Tenure (months)", min_value=0, max_value=12, value=12)

# Predict button
if st.button("🔍 Predict Segment"):

    # Skewed columns to log transform
    skewed_cols = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES',
                   'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                   'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

    # Build input dataframe
    input_data = {
        'BALANCE': BALANCE, 'PURCHASES': PURCHASES,
        'ONEOFF_PURCHASES': ONEOFF_PURCHASES,
        'INSTALLMENTS_PURCHASES': INSTALLMENTS_PURCHASES,
        'CASH_ADVANCE': CASH_ADVANCE, 'CREDIT_LIMIT': CREDIT_LIMIT,
        'PAYMENTS': PAYMENTS, 'MINIMUM_PAYMENTS': MINIMUM_PAYMENTS,
        'BALANCE_FREQUENCY': BALANCE_FREQUENCY,
        'PURCHASES_FREQUENCY': PURCHASES_FREQUENCY,
        'ONEOFF_PURCHASES_FREQUENCY': ONEOFF_PURCHASES_FREQUENCY,
        'PURCHASES_INSTALLMENTS_FREQUENCY': PURCHASES_INSTALLMENTS_FREQUENCY,
        'CASH_ADVANCE_FREQUENCY': CASH_ADVANCE_FREQUENCY,
        'CASH_ADVANCE_TRX': CASH_ADVANCE_TRX,
        'PURCHASES_TRX': PURCHASES_TRX,
        'PRC_FULL_PAYMENT': PRC_FULL_PAYMENT,
        'TENURE': TENURE
    }

    df_input = pd.DataFrame([input_data])

    # Apply log transformation
    for col in skewed_cols:
        df_input[col + '_LOG'] = np.log1p(df_input[col])
    df_input = df_input.drop(columns=skewed_cols)

    # Scale
    scaled = scaler.transform(df_input)

    # Predict
    cluster = kmeans.predict(scaled)[0]
    name, description = cluster_names[cluster]

    # Show result
    st.success(f"### 🎯 Cluster {cluster}: {name}")
    st.info(f"📋 {description}")


def add_background(image_url):
    st.markdown(
        f"""
        <style>
        /* Remove black top bar */
        header[data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0) !important;
        }}

        /* Background with blur */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            filter: blur(4px);
            z-index: -1;
        }}

        /* Dark overlay so text is readable */
        .stApp {{
            background: rgba(0, 0, 0, 0.45);
        }}

        /* Make all text white */
        h1, h2, h3, p, label, div {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with any image URL you like
add_background("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=1920")

#running command
#streamlit run "C:\Users\xnovaq.16\Documents\myenv\Marketing_Project\Marketing_analysis\app.py"