import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- Constants ---
DATA_FILE = "Telco2.csv"
MODEL_FILE = "svm_bundle.joblib"
FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

st.set_page_config(page_title="Churn quick-predict", layout="centered")
st.title("Churn quick-predict")

# --- Load and clean data ---
if not os.path.exists(DATA_FILE):
    st.error(f"CSV file `{DATA_FILE}` not found. Please make sure it's in the app folder.")
    st.stop()

df = pd.read_csv(DATA_FILE)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=FEATURES + ['Churn'], inplace=True)

X = df[FEATURES]
y = df['Churn'].astype(int)

# --- Load or train model ---
if os.path.exists(MODEL_FILE):
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    scaler = bundle["scaler"]
else:
    # Train and save model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel='rbf', C=100, gamma=1, probability=True, random_state=50)
    model.fit(X_train, y_train)

    # Save model and scaler in one file
    joblib.dump({"model": model, "scaler": scaler}, MODEL_FILE)
    st.success("✅ Model trained and saved.")

# --- UI: Input fields ---
st.header("Enter customer features")
col1, col2, col3 = st.columns(3)
with col1:
    tenure_val = st.number_input("tenure (months)", min_value=0.0, step=1.0, format="%.0f", value=12.0)
with col2:
    monthly_val = st.number_input("MonthlyCharges", min_value=0.0, step=0.1, value=70.0)
with col3:
    total_val = st.number_input("TotalCharges", min_value=0.0, step=0.1, value=800.0)

if st.button("Predict"):
    sample = pd.DataFrame([[tenure_val, monthly_val, total_val]], columns=FEATURES)
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    proba = model.predict_proba(sample_scaled)[0][1]

    if pred == 1:
        st.markdown("## 🔴 Likely to churn")
    else:
        st.markdown("## 🟢 Unlikely to churn")

    st.write(f"Predicted probability of churn: **{proba:.4f}**")
