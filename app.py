import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------ THEME ------------------
st.sidebar.title("🎨 Appearance")
theme = st.sidebar.radio("Choose Theme", ["🌙 Dark Mode", "☀ Light Mode"])
if theme == "☀ Light Mode":
    bg = "#f5f7fb"
    text = "#1f2937"
    card = "rgba(255,255,255,0.9)"
else:
    bg = "#0f172a"
    text = "#e2e8f0"
    card = "rgba(255,255,255,0.05)"

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
section[data-testid="stSidebar"] {{
background:#111827;
}}
section[data-testid="stSidebar"] * {{
color:white !important;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="main-title">📊 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict telecom customer churn using Machine Learning</div>', unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except:
    st.error("⚠ Model files not found")
    st.stop()

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.title("📋 Customer Details")

# Numeric
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 0, 200, 70)
total = st.sidebar.slider("Total Charges ($)", 0, 10000, 1500)

# Binary Categorical
def yes_no_to_binary(value):
    return 1 if value=="Yes" else 0

gender = st.sidebar.radio("Gender", ["Male", "Female"])
gender = 1 if gender=="Male" else 0

senior = st.sidebar.radio("Senior Citizen", [0, 1])
partner = yes_no_to_binary(st.sidebar.radio("Has Partner?", ["Yes","No"]))
dependents = yes_no_to_binary(st.sidebar.radio("Has Dependents?", ["Yes","No"]))
phoneservice = yes_no_to_binary(st.sidebar.radio("Phone Service?", ["Yes","No"]))
paperlessbilling = yes_no_to_binary(st.sidebar.radio("Paperless Billing?", ["Yes","No"]))

# Multiple options
multiplelines = st.sidebar.selectbox("Multiple Lines", ["No phone service","Yes","No"])
internetservice = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])
onlinesecurity = st.sidebar.selectbox("Online Security", ["Yes","No","No internet service"])
onlinebackup = st.sidebar.selectbox("Online Backup", ["Yes","No","No internet service"])
deviceprotection = st.sidebar.selectbox("Device Protection", ["Yes","No","No internet service"])
techsupport = st.sidebar.selectbox("Tech Support", ["Yes","No","No internet service"])
streamingtv = st.sidebar.selectbox("Streaming TV", ["Yes","No","No internet service"])
streamingmovies = st.sidebar.selectbox("Streaming Movies", ["Yes","No","No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month","One year","Two year"])
paymentmethod = st.sidebar.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

# ------------------ BUILD INPUT DF ------------------
input_dict = {
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phoneservice,
    'PaperlessBilling': paperlessbilling
}

# One-hot encode categorical variables (match training columns)
def encode_option(feature, value):
    encoding = {}
    for col in columns:
        if col.startswith(f"{feature}_"):
            encoding[col] = 1 if col == f"{feature}_{value}" else 0
    return encoding

for feature, value in {
    "MultipleLines": multiplelines,
    "InternetService": internetservice,
    "OnlineSecurity": onlinesecurity,
    "OnlineBackup": onlinebackup,
    "DeviceProtection": deviceprotection,
    "TechSupport": techsupport,
    "StreamingTV": streamingtv,
    "StreamingMovies": streamingmovies,
    "Contract": contract,
    "PaymentMethod": paymentmethod
}.items():
    input_dict.update(encode_option(feature, value))

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# Make sure all columns are present
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

# Scale numeric features
scaled = scaler.transform(input_df)

# ------------------ PREDICTION ------------------
st.markdown("### 🔍 Predict Customer Churn")

if st.button("Predict Churn"):
    with st.spinner("Analyzing customer data..."):
        time.sleep(2)
    prob = model.predict_proba(scaled)[0][1]
    prediction = model.predict(scaled)[0]

    col1,col2 = st.columns(2)
    with col1:
        if prediction==1:
            st.error("⚠ Customer WILL likely CHURN")
            st.snow()
        else:
            st.success("✅ Customer will likely STAY")
            st.balloons()
    with col2:
        st.metric("Churn Probability", f"{prob*100:.2f}%")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text':"Customer Churn Risk"},
        gauge={'axis':{'range':[0,100]},
               'steps':[{'range':[0,50],'color':'green'},
                        {'range':[50,80],'color':'orange'},
                        {'range':[80,100],'color':'red'}]}
    ))
    st.plotly_chart(fig,use_container_width=True)

# ------------------ FEATURE IMPORTANCE ------------------
st.markdown("### 📊 Feature Importance")
try:
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"Feature":columns,"Importance":importance}).sort_values("Importance")
    fig = px.bar(
        imp_df,x="Importance",y="Feature",orientation="h",
        color="Importance",
        template="plotly_white" if theme=="☀ Light Mode" else "plotly_dark"
    )
    st.plotly_chart(fig,use_container_width=True)
except:
    st.info("Feature importance unavailable")

# ------------------ HEATMAP ------------------
st.markdown("### 📊 Customer Data Correlation")
try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    corr = df.select_dtypes(include="number").corr()
    fig = px.imshow(corr,text_auto=True,color_continuous_scale="RdBu",
                    template="plotly_white" if theme=="☀ Light Mode" else "plotly_dark")
    st.plotly_chart(fig,use_container_width=True)
except:
    st.warning("Dataset not found")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("""
### 🌐 About
This dashboard helps businesses **identify customers at risk of leaving** using predictive analytics.

👩‍💻 Developed by **Tibah Wajahat**
""")
