import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
    }

    .result-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    }
    .result-box h1 { color: white; font-size: 3rem; margin: 0; }
    .result-box p  { color: rgba(255,255,255,0.8); font-size: 1.1rem; margin: 8px 0 0; }

    .metric-card {
        background: rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
    .metric-card .label { font-size: 0.8rem; color: rgba(255,255,255,0.5); margin-top: 4px; }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green  { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
    .badge-yellow { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-red    { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid #f87171; }

    h1, h2, h3 { color: white !important; }
    p, label, .stMarkdown { color: rgba(255,255,255,0.85) !important; }

    .stNumberInput input, .stSelectbox select {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.6) !important;
    }

    hr { border-color: rgba(255,255,255,0.1) !important; }
    .stTabs [data-baseweb="tab"] { color: rgba(255,255,255,0.6) !important; }
    .stTabs [aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model & data ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("car_price_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("car.csv")

model = load_model()
df    = load_data()

CURRENT_YEAR = 2024

# ── Sidebar inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Car Price Predictor")
    st.markdown("Powered by **Gradient Boosting** · R² = 0.976")
    st.divider()

    st.markdown("### 📋 Car Details")

    year = st.slider("Manufacturing Year", min_value=1990, max_value=CURRENT_YEAR, value=2015)
    car_age = CURRENT_YEAR - year
    st.caption(f"Car age: **{car_age} year(s)**")

    present_price = st.number_input(
        "Showroom Price (₹ Lakhs)",
        min_value=0.1, max_value=200.0, value=5.0, step=0.1,
        help="Current ex-showroom price of this car model"
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0, max_value=1_000_000, value=50000, step=1000
    )

    st.markdown("### ⚙️ Specifications")

    fuel_type    = st.selectbox("Fuel Type",    ["Petrol", "Diesel", "CNG"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    seller_type  = st.selectbox("Seller Type",  ["Dealer", "Individual"])
    owner        = st.selectbox("Previous Owners", [0, 1, 3],
                                format_func=lambda x: ["First owner", "Second owner", "Third owner+"][{0:0,1:1,3:2}[x]])

    st.divider()
    predict_btn = st.button("🔮 Predict Selling Price", use_container_width=True)

# ── Encode ───────────────────────────────────────────────────────────────────
fuel_map         = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map       = {"Dealer": 0, "Individual": 1}
transmission_map = {"Manual": 0, "Automatic": 1}

input_df = pd.DataFrame([[
    year, car_age, present_price, kms_driven,
    fuel_map[fuel_type], seller_map[seller_type],
    transmission_map[transmission], owner
]], columns=["Year", "Car_Age", "Present_Price", "Kms_Driven",
             "Fuel_Type", "Seller_Type", "Transmission", "Owner"])

# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 Car Price Predictor")
st.markdown("Instant resale value estimation using Gradient Boosting — R² score of **0.976** on test data.")
st.divider()

tab1, tab2, tab3 = st.tabs(["💰 Prediction", "📊 Data Insights", "ℹ️ About"])

# ── Tab 1 ────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Your Car Summary")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="value">{car_age}y</div><div class="label">Car Age</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="value">₹{present_price}L</div><div class="label">Showroom</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="value">{kms_driven:,}</div><div class="label">KMs</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Condition badges
        condition_notes = []
        if car_age <= 3:
            condition_notes.append('<span class="badge badge-green">Nearly New</span>')
        elif car_age <= 7:
            condition_notes.append('<span class="badge badge-yellow">Moderate Age</span>')
        else:
            condition_notes.append('<span class="badge badge-red">High Age</span>')

        if kms_driven < 30000:
            condition_notes.append('<span class="badge badge-green">Low Mileage</span>')
        elif kms_driven < 80000:
            condition_notes.append('<span class="badge badge-yellow">Average Mileage</span>')
        else:
            condition_notes.append('<span class="badge badge-red">High Mileage</span>')

        if owner == 0:
            condition_notes.append('<span class="badge badge-green">First Owner</span>')
        elif owner == 1:
            condition_notes.append('<span class="badge badge-yellow">Second Owner</span>')
        else:
            condition_notes.append('<span class="badge badge-red">Third Owner+</span>')

        st.markdown(" ".join(condition_notes), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        for k, v in [("Fuel", fuel_type), ("Transmission", transmission), ("Seller", seller_type)]:
            st.markdown(f"**{k}:** {v}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if predict_btn:
            prediction = max(0.0, model.predict(input_df)[0])

            st.markdown(f"""
            <div class="result-box">
                <p>Estimated Resale Value</p>
                <h1>₹ {prediction:.2f} L</h1>
                <p>Gradient Boosting · R² = 0.976 · MAE ≈ ₹0.49L</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            retention = (prediction / present_price * 100) if present_price > 0 else 0
            depreciation = max(0, present_price - prediction)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 Value Analysis")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f'<div class="metric-card"><div class="value">{retention:.1f}%</div><div class="label">Value Retained</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="metric-card"><div class="value">₹{depreciation:.2f}L</div><div class="label">Depreciation</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(min(retention / 100, 1.0), text=f"{retention:.1f}% of showroom price retained")
            st.markdown('</div>', unsafe_allow_html=True)

            # Price range estimate (±MAE)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📉 Confidence Range")
            st.markdown(f"Based on model MAE of ~₹0.49L, the realistic range is:")
            st.markdown(f"**₹ {max(0, prediction-0.49):.2f}L  —  ₹ {prediction+0.49:.2f}L**")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 60px 24px;">
                <div style="font-size: 4rem;">�</div>
                <h3>Ready to Predict</h3>
                <p>Fill in the car details in the sidebar and click <strong>Predict Selling Price</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{len(df)}</div><div class="label">Records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="value">₹{df["Selling_Price"].mean():.1f}L</div><div class="label">Avg Price</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="value">₹{df["Selling_Price"].max():.0f}L</div><div class="label">Max Price</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="value">{df["Year"].min()}–{df["Year"].max()}</div><div class="label">Year Range</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Fuel Type Distribution**")
        st.bar_chart(df["Fuel_Type"].value_counts())
    with col_b:
        st.markdown("**Transmission Distribution**")
        st.bar_chart(df["Transmission"].value_counts())

    st.markdown("**Average Selling Price by Year**")
    st.line_chart(df.groupby("Year")["Selling_Price"].mean())

    st.markdown("**Feature Importance (Gradient Boosting)**")
    importance = {
        "Present_Price": 0.8835, "Year": 0.0334, "Car_Age": 0.0474,
        "Kms_Driven": 0.0238, "Fuel_Type": 0.0084, "Transmission": 0.0035,
        "Seller_Type": 0.0, "Owner": 0.0
    }
    imp_df = pd.DataFrame(importance.items(), columns=["Feature", "Importance"]).sort_values("Importance")
    st.bar_chart(imp_df.set_index("Feature"))

    with st.expander("🔍 View Raw Dataset"):
        st.dataframe(df, use_container_width=True)

# ── Tab 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🤖 About This Project

    Predicts the **resale price of a used car** using a **Gradient Boosting Regressor**
    trained on 301 real car listings.

    **Model Performance:**
    - R² Score: **0.976** (test set)
    - MAE: **~₹0.49 Lakhs**
    - RMSE: **~₹0.74 Lakhs**

    **Why Gradient Boosting over Lasso?**
    The original Lasso model (alpha=1.0) zeroed out 4 of 7 features due to over-regularization,
    effectively ignoring Fuel Type, Seller Type, Transmission, and Owner. Gradient Boosting
    captures non-linear relationships and achieves significantly better accuracy.

    **Features used:**
    - Year & Car Age · Present Showroom Price · Kilometers Driven
    - Fuel Type · Transmission · Seller Type · Previous Owners

    **Tech Stack:** Python · Scikit-learn · Streamlit · Pandas · NumPy
    """)
    st.markdown('</div>', unsafe_allow_html=True)
