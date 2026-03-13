import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import time
from datetime import datetime

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #050d1a;
  --surface: #0b1628;
  --surface2: #112240;
  --border: #1e3a5f;
  --accent: #00d4ff;
  --accent2: #7c3aed;
  --green: #10b981;
  --red: #ef4444;
  --amber: #f59e0b;
  --text: #ccd6f6;
  --muted: #8892b0;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
    display: block;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,212,255,0.08);
}

/* Remove default padding */
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

/* Headings */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #e6f1ff !important; }

/* Cards */
.ci-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.ci-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* KPI boxes */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }
.kpi-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-box:hover { transform: translateY(-2px); border-color: var(--accent); }
.kpi-number { font-size: 32px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 4px 0; }
.kpi-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; }
.kpi-accent { color: var(--accent); }
.kpi-red { color: var(--red); }
.kpi-green { color: var(--green); }
.kpi-amber { color: var(--amber); }

/* Hero header */
.hero-title {
    font-size: 52px;
    font-weight: 700;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #e6f1ff 0%, #00d4ff 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 8px;
}
.hero-sub {
    font-size: 16px;
    color: var(--muted);
    letter-spacing: 0.5px;
    margin-bottom: 32px;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.badge-risk { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.badge-safe { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }

/* Prediction result */
.pred-box {
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    border: 1px solid;
}
.pred-churn {
    background: rgba(239,68,68,0.07);
    border-color: rgba(239,68,68,0.4);
}
.pred-stay {
    background: rgba(16,185,129,0.07);
    border-color: rgba(16,185,129,0.4);
}
.pred-title { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
.pred-prob { font-size: 48px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* Sidebar logo */
.sidebar-logo {
    text-align: center;
    padding: 20px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.sidebar-logo-text {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #e6f1ff !important;
}
.sidebar-logo-accent { color: var(--accent) !important; }

/* Nav item */
.nav-label {
    font-size: 11px;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 20px 0 8px;
    padding-left: 4px;
}

/* Table */
.styled-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.styled-table th {
    background: var(--surface2);
    color: var(--muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 11px;
    padding: 10px 14px;
    text-align: left;
}
.styled-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.styled-table tr:hover td { background: rgba(0,212,255,0.03); }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.5px !important;
    height: 44px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Sliders + inputs */
.stSlider, .stSelectbox, .stNumberInput { }
.stSlider [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }
label { color: var(--muted) !important; font-size: 13px !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* Expander */
.streamlit-expanderHeader { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text) !important; }

/* Info / warning overrides */
.stAlert { border-radius: 10px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 500 !important; border-bottom: 2px solid transparent !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    return df

try:
    model, scaler, columns = load_model()
    df_raw = load_data()
    assets_ok = True
except Exception as e:
    assets_ok = False
    st.error(f"⚠ Could not load model files: {e}")

# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("churn_history.db", check_same_thread=False)
    # Check existing columns and migrate if needed
    cursor = conn.execute("PRAGMA table_info(predictions)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if not existing_cols:
        # Table doesn't exist yet — create fresh
        conn.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                tenure INTEGER,
                monthly_charges REAL,
                total_charges REAL,
                probability REAL,
                prediction TEXT
            )
        """)
        conn.commit()
    elif "prediction" not in existing_cols or "probability" not in existing_cols:
        # Old schema detected — drop and recreate
        conn.execute("DROP TABLE predictions")
        conn.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                tenure INTEGER,
                monthly_charges REAL,
                total_charges REAL,
                probability REAL,
                prediction TEXT
            )
        """)
        conn.commit()
    return conn

def save_prediction(tenure, monthly, total, prob, pred):
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (timestamp, tenure, monthly_charges, total_charges, probability, prediction) VALUES (?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tenure, monthly, total, round(prob*100, 2), pred)
    )
    conn.commit()
    conn.close()

def get_history():
    conn = get_db()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    return df

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-logo">
  <div style="font-size:32px; margin-bottom:6px;">⚡</div>
  <div class="sidebar-logo-text">Churn <span class="sidebar-logo-accent">Intelligence</span></div>
  <div style="font-size:11px; color:#8892b0; margin-top:4px;">Telecom Analytics Platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["🏠  Home", "🔮  Predict", "📊  Dashboard", "🗄  Data", "📜  History"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="nav-label">About</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="font-size:12px; color:#8892b0; line-height:1.8; padding: 0 4px;">
👩‍💻 <b style="color:#ccd6f6;">Tibah Wajahat</b><br>
🧠 Model: Random Forest<br>
📦 Dataset: IBM Telco<br>
🔧 Stack: Python · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<div class="hero-title">Churn Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-powered customer retention analytics for telecom providers</div>', unsafe_allow_html=True)

    # KPIs from dataset
    if assets_ok:
        total_customers = len(df_raw)
        churned = df_raw["Churn_Binary"].sum()
        churn_rate = churned / total_customers * 100
        avg_tenure = df_raw["tenure"].mean()
        avg_monthly = df_raw["MonthlyCharges"].mean()

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-label">Total Customers</div>
                <div class="kpi-number kpi-accent">{total_customers:,}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Churn Rate</div>
                <div class="kpi-number kpi-red">{churn_rate:.1f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Avg Tenure</div>
                <div class="kpi-number kpi-amber">{avg_tenure:.0f} mo</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Avg Monthly</div>
                <div class="kpi-number kpi-green">${avg_monthly:.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### 📌 What is Churn Intelligence?")
        st.markdown("""
Churn Intelligence is a **machine learning dashboard** built to help telecom providers proactively identify customers at risk of cancelling their service.

Using a **Random Forest classifier** trained on IBM's Telco dataset, the platform delivers:

- 🔮 **Real-time churn predictions** with probability scores
- 📊 **Interactive dashboards** with key churn drivers
- 🗄 **Data exploration** across customer segments
- 📜 **Prediction history** log with SQLite persistence
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Quick Navigation")
        if st.button("🔮 Make a Prediction", use_container_width=True):
            st.session_state["_nav"] = "predict"
            st.rerun()
        st.markdown("&nbsp;")
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state["_nav"] = "dashboard"
            st.rerun()
        st.markdown("&nbsp;")
        if st.button("📜 Prediction History", use_container_width=True):
            st.session_state["_nav"] = "history"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if assets_ok:
        # Churn by contract type
        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### Churn Rate by Contract Type")
        contract_churn = df_raw.groupby("Contract")["Churn_Binary"].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        contract_churn["Churn Rate"] *= 100
        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate", color="Churn Rate",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            template="plotly_dark", text_auto=".1f"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk", showlegend=False,
            yaxis_title="Churn Rate (%)", coloraxis_showscale=False,
            margin=dict(t=10, b=10)
        )
        fig.update_traces(textposition="outside", textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("## 🔮 Churn Prediction")
    st.markdown('<p style="color:#8892b0;">Enter customer details to predict churn probability.</p>', unsafe_allow_html=True)

    if not assets_ok:
        st.error("Model files could not be loaded.")
        st.stop()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### Customer Profile")

        tenure = st.slider("📅 Tenure (Months)", 0, 72, 12)
        monthly = st.slider("💵 Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
        total = st.number_input("🧾 Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly))

        st.markdown("---")
        predict_btn = st.button("⚡ Predict Churn Risk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="ci-card" style="min-height:340px;">', unsafe_allow_html=True)
        st.markdown("#### Prediction Result")

        if predict_btn:
            with st.spinner("Running model inference..."):
                time.sleep(1.2)

            input_df = pd.DataFrame({"tenure": [tenure], "MonthlyCharges": [monthly], "TotalCharges": [total]})
            for col in columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[columns]
            scaled = scaler.transform(input_df)
            prob = model.predict_proba(scaled)[0][1]
            pred_label = "CHURN" if prob > 0.5 else "STAY"

            # Save to history
            save_prediction(tenure, monthly, total, prob, pred_label)

            # Show result
            box_class = "pred-churn" if pred_label == "CHURN" else "pred-stay"
            icon = "⚠️" if pred_label == "CHURN" else "✅"
            color = "#ef4444" if pred_label == "CHURN" else "#10b981"
            badge_class = "badge-risk" if pred_label == "CHURN" else "badge-safe"
            msg = "High churn risk detected" if pred_label == "CHURN" else "Customer likely to stay"

            st.markdown(f"""
            <div class="pred-box {box_class}">
                <div class="pred-title">{icon} {msg}</div>
                <div class="pred-prob" style="color:{color};">{prob*100:.1f}%</div>
                <div style="color:#8892b0; font-size:13px; margin-top:4px;">Churn Probability</div>
                <div style="margin-top:12px;"><span class="badge {badge_class}">{pred_label}</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"family": "JetBrains Mono", "size": 28, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16,185,129,0.15)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob * 100}
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6", height=200, margin=dict(t=20, b=0, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; color:#8892b0; padding:60px 20px;">
                <div style="font-size:48px; margin-bottom:12px;">🔮</div>
                <div>Fill in the customer profile and click <b>Predict</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="ci-card">', unsafe_allow_html=True)
    st.markdown("#### Feature Importance")
    try:
        imp_df = pd.DataFrame({"Feature": columns, "Importance": model.feature_importances_})
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#1e3a5f", "#00d4ff"],
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk", coloraxis_showscale=False,
            yaxis_title="", margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance unavailable for this model type.")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown('<p style="color:#8892b0;">Explore churn patterns across the customer base.</p>', unsafe_allow_html=True)

    if not assets_ok:
        st.error("Dataset not loaded.")
        st.stop()

    tabs = st.tabs(["📈 Churn Overview", "💰 Revenue Impact", "🌐 Correlation"])

    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="ci-card">', unsafe_allow_html=True)
            st.markdown("#### Churn Distribution")
            churn_counts = df_raw["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Status", "Count"]
            fig = px.pie(
                churn_counts, values="Count", names="Status",
                color_discrete_sequence=["#10b981", "#ef4444"],
                hole=0.55, template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_family="Space Grotesk", margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="ci-card">', unsafe_allow_html=True)
            st.markdown("#### Tenure vs. Churn")
            fig = px.histogram(
                df_raw, x="tenure", color="Churn",
                color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
                nbins=30, barmode="overlay", opacity=0.75, template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Space Grotesk", margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### Monthly Charges Distribution by Churn")
        fig = px.violin(
            df_raw, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
            box=True, points="outliers", template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Space Grotesk", showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="ci-card">', unsafe_allow_html=True)
            st.markdown("#### Avg Monthly Charges: Churn vs Retained")
            avg_charge = df_raw.groupby("Churn")["MonthlyCharges"].mean().reset_index()
            fig = px.bar(
                avg_charge, x="Churn", y="MonthlyCharges",
                color="Churn", color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
                template="plotly_dark", text_auto=".1f"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Space Grotesk", showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="ci-card">', unsafe_allow_html=True)
            st.markdown("#### Churn Rate by Internet Service")
            if "InternetService" in df_raw.columns:
                inet_churn = df_raw.groupby("InternetService")["Churn_Binary"].mean().reset_index()
                inet_churn["Churn Rate"] = inet_churn["Churn_Binary"] * 100
                fig = px.bar(
                    inet_churn, x="InternetService", y="Churn Rate",
                    color="Churn Rate", color_continuous_scale=["#10b981", "#ef4444"],
                    template="plotly_dark", text_auto=".1f"
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Space Grotesk", coloraxis_showscale=False, showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### Total Charges vs Tenure (colored by Churn)")
        fig = px.scatter(
            df_raw.sample(min(1000, len(df_raw))), x="tenure", y="TotalCharges",
            color="Churn", color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
            opacity=0.6, template="plotly_dark", size_max=6
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Space Grotesk", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### Feature Correlation Heatmap")
        corr = df_raw.select_dtypes(include="number").corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            template="plotly_dark", aspect="auto"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_family="Space Grotesk", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: DATA
# ═══════════════════════════════════════════════════════
elif "Data" in page:
    st.markdown("## 🗄 Customer Data Explorer")
    st.markdown('<p style="color:#8892b0;">Browse and filter the telecom dataset.</p>', unsafe_allow_html=True)

    if not assets_ok:
        st.error("Dataset not loaded.")
        st.stop()

    # Filters
    st.markdown('<div class="ci-card">', unsafe_allow_html=True)
    st.markdown("#### Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        churn_filter = st.selectbox("Churn Status", ["All", "Yes", "No"])
    with col2:
        contract_filter = st.selectbox("Contract Type", ["All"] + df_raw["Contract"].unique().tolist())
    with col3:
        tenure_range = st.slider("Tenure Range (Months)", 0, 72, (0, 72))
    st.markdown('</div>', unsafe_allow_html=True)

    filtered = df_raw.copy()
    if churn_filter != "All":
        filtered = filtered[filtered["Churn"] == churn_filter]
    if contract_filter != "All":
        filtered = filtered[filtered["Contract"] == contract_filter]
    filtered = filtered[(filtered["tenure"] >= tenure_range[0]) & (filtered["tenure"] <= tenure_range[1])]

    st.markdown(f'<p style="color:#8892b0; font-size:13px;">Showing <b style="color:#00d4ff;">{len(filtered):,}</b> records</p>', unsafe_allow_html=True)

    st.markdown('<div class="ci-card">', unsafe_allow_html=True)
    display_cols = ["customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "InternetService", "Churn"]
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available_cols].reset_index(drop=True),
        use_container_width=True,
        height=420
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Summary stats
    st.markdown('<div class="ci-card">', unsafe_allow_html=True)
    st.markdown("#### Summary Statistics")
    st.dataframe(filtered.select_dtypes(include="number").describe().round(2), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: HISTORY
# ═══════════════════════════════════════════════════════
elif "History" in page:
    st.markdown("## 📜 Prediction History")
    st.markdown('<p style="color:#8892b0;">All past churn predictions made in this session.</p>', unsafe_allow_html=True)

    hist_df = get_history()

    if hist_df.empty:
        st.markdown("""
        <div style="text-align:center; padding:80px 0; color:#8892b0;">
            <div style="font-size:48px; margin-bottom:12px;">📭</div>
            <div>No predictions yet. Head to the <b>Predict</b> page to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # KPIs
        total_preds = len(hist_df)
        churn_preds = (hist_df["prediction"] == "CHURN").sum()
        avg_prob = hist_df["probability"].mean()

        st.markdown(f"""
        <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr);">
            <div class="kpi-box">
                <div class="kpi-label">Total Predictions</div>
                <div class="kpi-number kpi-accent">{total_preds}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Churn Alerts</div>
                <div class="kpi-number kpi-red">{churn_preds}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Avg Risk Score</div>
                <div class="kpi-number kpi-amber">{avg_prob:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ci-card">', unsafe_allow_html=True)

        # Color prediction column
        def color_pred(val):
            return "color: #ef4444; font-weight:600;" if val == "CHURN" else "color: #10b981; font-weight:600;"

        styled = hist_df[["timestamp", "tenure", "monthly_charges", "total_charges", "probability", "prediction"]].rename(
            columns={
                "timestamp": "Timestamp", "tenure": "Tenure", "monthly_charges": "Monthly ($)",
                "total_charges": "Total ($)", "probability": "Probability (%)", "prediction": "Prediction"
            }
        )
        st.dataframe(
            styled.style.applymap(color_pred, subset=["Prediction"]),
            use_container_width=True,
            height=400
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Trend chart
        st.markdown('<div class="ci-card">', unsafe_allow_html=True)
        st.markdown("#### Churn Probability Over Predictions")
        hist_df["index"] = range(len(hist_df) - 1, -1, -1)
        fig = px.line(
            hist_df.sort_values("index"), x="index", y="probability",
            color="prediction", color_discrete_map={"CHURN": "#ef4444", "STAY": "#10b981"},
            markers=True, template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk", xaxis_title="Prediction #", yaxis_title="Probability (%)",
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑 Clear History", use_container_width=True):
                conn = get_db()
                conn.execute("DELETE FROM predictions")
                conn.commit()
                conn.close()
                st.rerun()
        with col2:
            csv = styled.to_csv(index=False)
            st.download_button("⬇ Download CSV", csv, "churn_history.csv", "text/csv", use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-size:12px; padding:10px 0;">
    ⚡ <b style="color:#ccd6f6;">Churn Intelligence</b> · Built with Streamlit & Scikit-Learn · 
    👩‍💻 Tibah Wajahat
</div>
""", unsafe_allow_html=True)
