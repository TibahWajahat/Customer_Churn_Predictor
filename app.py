import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# THEME SWITCH
# ------------------------------------------------
st.sidebar.title("🎨 Appearance")

theme = st.sidebar.radio(
    "Choose Theme",
    ["🌙 Dark Mode", "☀ Light Mode"]
)

if theme == "☀ Light Mode":
    bg = "#f5f7fb"
    text = "#1f2937"
    card = "rgba(255,255,255,0.9)"
else:
    bg = "#0f172a"
    text = "#e2e8f0"
    card = "rgba(255,255,255,0.05)"

# ------------------------------------------------
# GLOBAL STYLING
# ------------------------------------------------
st.markdown(f"""
<style>

.stApp {{
background-color: {bg};
color:{text};
}}

.main-title {{
font-size:50px;
font-weight:800;
text-align:center;
margin-bottom:10px;
}}

.subtitle {{
text-align:center;
font-size:20px;
margin-bottom:30px;
opacity:0.8;
}}

.card {{
background:{card};
padding:30px;
border-radius:14px;
border:1px solid rgba(255,255,255,0.08);
backdrop-filter: blur(10px);
margin-bottom:20px;
}}

.metric-box {{
background:{card};
padding:20px;
border-radius:12px;
text-align:center;
border:1px solid rgba(255,255,255,0.08);
}}

.stButton>button {{
background:#6366f1;
color:white;
border-radius:8px;
border:none;
height:40px;
font-weight:600;
}}

section[data-testid="stSidebar"] {{
background:#111827;
}}

section[data-testid="stSidebar"] * {{
color:white !important;
}}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.markdown(
'<div class="main-title">📊 Customer Churn Prediction Dashboard</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Predict telecom customer churn using Machine Learning</div>',
unsafe_allow_html=True
)

# ------------------------------------------------
# PROJECT INFO
# ------------------------------------------------
st.markdown(f"""
<div class="card">

### 👩‍💻 Project Information

This interactive dashboard predicts **telecom customer churn probability** using a machine learning model.

**Technologies used**

• Python  
• Scikit-Learn  
• Streamlit  
• Plotly  

⭐ Developed by **Tibah Wajahat**

</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except:
    st.error("⚠ Model files not found")
    st.stop()

# ------------------------------------------------
# SIDEBAR INPUT
# ------------------------------------------------
st.sidebar.title("📋 Customer Details")

tenure = st.sidebar.slider("Tenure (Months)",0,72,12)
monthly = st.sidebar.slider("Monthly Charges ($)",0,200,70)
total = st.sidebar.slider("Total Charges ($)",0,10000,1500)

# ------------------------------------------------
# INPUT DATA
# ------------------------------------------------
input_df = pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
})

for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

scaled = scaler.transform(input_df)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
st.markdown("### 🔍 Predict Customer Churn")

if st.button("Predict Churn"):

    with st.spinner("Analyzing customer data..."):
        time.sleep(2)

    prob = model.predict_proba(scaled)[0][1]

    col1, col2 = st.columns(2)

    with col1:

        if prob > 0.5:
            st.error("⚠ Customer likely to churn")
            st.snow()
        else:
            st.success("✅ Customer likely to stay")
            st.balloons()

    with col2:

        st.metric(
            "Churn Probability",
            f"{prob*100:.2f}%"
        )

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text':"Churn Risk"},
        gauge={'axis':{'range':[0,100]}}
    ))

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
st.markdown("### 📊 Feature Importance")

try:

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature":columns,
        "Importance":importance
    }).sort_values("Importance")

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        template="plotly_white" if theme=="☀ Light Mode" else "plotly_dark"
    )

    st.plotly_chart(fig,use_container_width=True)

except:
    st.info("Feature importance unavailable")

# ------------------------------------------------
# HEATMAP
# ------------------------------------------------
st.markdown("### 📊 Customer Data Correlation")

try:

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        template="plotly_white" if theme=="☀ Light Mode" else "plotly_dark"
    )

    st.plotly_chart(fig,use_container_width=True)

except:
    st.warning("Dataset not found")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")

st.markdown(
"""
### 🌐 About

This dashboard helps businesses **identify customers at risk of leaving** by using predictive analytics.

👩‍💻 Developed by **Tibah Wajahat**
"""
)
