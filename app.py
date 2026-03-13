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
    page_title="Customer Churn AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# THEME SWITCHER
# ------------------------------------------------
st.sidebar.title("🎨 Dashboard Theme")

theme = st.sidebar.radio(
    "Select Mode",
    ["🌙 Dark Mode", "☀ Light Mode"]
)

if theme == "☀ Light Mode":
    bg = "#f8fafc"
    text = "#111"
    card = "rgba(255,255,255,0.9)"
else:
    bg = "#020617"
    text = "white"
    card = "rgba(255,255,255,0.05)"

# ------------------------------------------------
# CUSTOM STYLE
# ------------------------------------------------
st.markdown(f"""
<style>

.stApp {{
background:{bg};
color:{text};
}}

.main-title {{
font-size:56px;
font-weight:900;
text-align:center;
background:linear-gradient(90deg,#ff2e9b,#ff7acb);
-webkit-background-clip:text;
color:transparent;
}}

.subtitle {{
text-align:center;
font-size:22px;
margin-bottom:30px;
}}

.card {{
background:{card};
padding:25px;
border-radius:16px;
border:1px solid rgba(255,255,255,0.1);
}}

.stButton>button {{
background:linear-gradient(90deg,#ff2e9b,#ff7acb);
color:white;
border:none;
border-radius:8px;
height:40px;
font-weight:600;
}}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.markdown(
'<p class="main-title">📊 Customer Churn Prediction Dashboard</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Predict telecom customer churn using Machine Learning</p>',
unsafe_allow_html=True
)

# ------------------------------------------------
# PROJECT INFO
# ------------------------------------------------
st.markdown(f"""
<div class="card">

### 👩‍💻 Project Information

Customer Churn Prediction AI Dashboard

Built with:

• Python  
• Machine Learning  
• Streamlit  
• Plotly  

⭐ Developed by **Tibah Wajahat**

</div>
""", unsafe_allow_html=True)

st.markdown("---")

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

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 200, 70)
total = st.sidebar.slider("Total Charges", 0, 10000, 1500)

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
if st.sidebar.button("🚀 Predict Churn"):

    st.toast("🤖 AI analyzing customer data...")

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    prob = model.predict_proba(scaled)[0][1]

    st.toast("✅ Prediction Complete")

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

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text':"Churn Risk"},
        gauge={'axis':{'range':[0,100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
st.header("📊 Feature Importance")

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
        color="Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

except:
    st.info("Feature importance unavailable")

# ------------------------------------------------
# HEATMAP
# ------------------------------------------------
st.header("📊 Customer Churn Heatmap")

try:

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu"
    )

    st.plotly_chart(fig, use_container_width=True)

except:
    st.warning("Dataset not found")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")

st.markdown("""
### 🌐 About

This dashboard predicts **telecom customer churn probability**
using machine learning models.

👩‍💻 Developed by **Tibah Wajahat**
""")
