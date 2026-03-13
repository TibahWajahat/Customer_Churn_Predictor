import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}
.card {
    background-color:#f5f5f5;
    padding:20px;
    border-radius:15px;
    box-shadow:2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown('<p class="main-title">📊 Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
st.write("Predict whether a telecom customer will churn using machine learning.")

# -----------------------------
# Load Model Safely
# -----------------------------
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except:
    st.error("⚠ Model files not found. Please ensure .pkl files are in the project folder.")
    st.stop()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Customer Information")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0, 200, 70)
total_charges = st.sidebar.slider("Total Charges", 0, 10000, 1500)

# -----------------------------
# Prepare Input Data
# -----------------------------
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])

# Ensure columns match training data
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

scaled_data = scaler.transform(input_df)

# -----------------------------
# Prediction Button
# -----------------------------
st.sidebar.markdown("---")

if st.sidebar.button("🔍 Predict Churn"):

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠ High Risk: Customer likely to churn")
        else:
            st.success("✅ Low Risk: Customer likely to stay")

    with col2:
        st.metric("Churn Probability", f"{probability*100:.2f}%")

    # -----------------------------
    # Gauge Chart
    # -----------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text': "Churn Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Feature Visualization
# -----------------------------
st.header("📈 Customer Feature Overview")

chart_df = pd.DataFrame({
    "Feature": ["Tenure", "Monthly Charges", "Total Charges"],
    "Value": [tenure, monthly_charges, total_charges]
})

fig = px.bar(
    chart_df,
    x="Feature",
    y="Value",
    color="Feature",
    title="Customer Input Data"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Dataset Analytics
# -----------------------------
st.header("📂 Dataset Insights")

try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Average Monthly Charges", round(df["MonthlyCharges"].mean(),2))
    col3.metric("Average Tenure", round(df["tenure"].mean(),2))

    pie = px.pie(df, names="Churn", title="Churn Distribution")
    st.plotly_chart(pie)

    hist = px.histogram(df, x="MonthlyCharges", nbins=30,
                        title="Monthly Charges Distribution")
    st.plotly_chart(hist)

except:
    st.info("Dataset file not found. Upload dataset to view analytics.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Customer Churn Prediction System | Built with Streamlit")