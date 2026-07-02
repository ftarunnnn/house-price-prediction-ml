"""
Phase 16 – Deployment
House Price Prediction
"""
import json
import os

import joblib
import streamlit as st

from src.predict import (
    FEATURES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    format_inr,
    predict_price,
)

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
HOUSE_PATH = os.path.join(BASE_DIR, "assets", "house.png")


# ── Cached artefacts (loaded once per session) ───────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model once and reuse it for every prediction."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    """Load the fitted scaler once and reuse it for every prediction."""
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_feature_columns():
    """Load feature column names once and reuse them for every prediction."""
    with open(FEATURES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(LOGO_PATH, width=120)
    st.markdown(
        """
        <p style="text-align:center; color:#6366f1; font-weight:700; margin-top:-0.5rem;">
        House Price Predictor
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Project Details")
    st.markdown(
        """
        **🎯 Goal:** Predict house selling price

        **🤖 Models tried:** Linear Regression, Decision Tree,
        Random Forest, Gradient Boosting

        **🏆 Best model:** Linear Regression (R² ≈ 0.97)
        """
    )

    st.divider()
    with st.expander("🚀 Deployment Guide", expanded=False):
        st.markdown(
            """
            **Deploy this app for free:**

            1️⃣ **Streamlit Cloud** *(Recommended)*
            - Push repo to GitHub
            - Visit [share.streamlit.io](https://share.streamlit.io)
            - Connect repo → set `app.py` → Deploy

            2️⃣ **Hugging Face Spaces**
            - New Space → SDK: Streamlit
            - Push repo contents

            3️⃣ **Render / Railway**
            - Start command:
            ```
            streamlit run app.py
            --server.port $PORT
            --server.address 0.0.0.0
            ```
            """
        )
    st.divider()
    st.markdown("### ⚙️ Input Features")

    area = st.number_input("📐 Area (sq ft)", min_value=500, max_value=10000, value=1800, step=50)
    bedrooms = st.slider("🛏️ Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("🚿 Bathrooms", min_value=1, max_value=6, value=2)
    garage = st.slider("🚗 Garage", min_value=0, max_value=4, value=1)
    age = st.number_input("📅 House Age (years)", min_value=0, max_value=100, value=10, step=1)
    location = st.selectbox("📍 Location", ["Urban", "Semi Urban", "Rural"])


# ── Main page ────────────────────────────────────────────────────────────────
hero_left, hero_right = st.columns([1.2, 1.8])

with hero_left:
    st.image(HOUSE_PATH, use_container_width=True, caption="Find your property value")

with hero_right:
    st.markdown(
        """
        <h1 style="
            background: linear-gradient(90deg, #6366f1, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        ">
            🏠 House Price Prediction
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Enter property details in the sidebar, then click **Predict Price** "
        "to get an instant ML-powered estimate."
    )

st.divider()

predict_col, summary_col = st.columns([1, 2])

with predict_col:
    predict_clicked = st.button("🔮 Predict Price", type="primary", use_container_width=True)

with summary_col:
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    with metric_a:
        st.metric("📐 Area", f"{area:,} sq ft")
    with metric_b:
        st.metric("🛏️ Bedrooms", bedrooms)
    with metric_c:
        st.metric("🚿 Bathrooms", bathrooms)
    with metric_d:
        st.metric("📍 Location", location)

if predict_clicked:
    try:
        predicted_price = predict_price(
            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            garage=garage,
            age=age,
            location=location,
            model=load_model(),
            scaler=load_scaler(),
            feature_cols=load_feature_columns(),
        )

        st.success("✅ Prediction complete! Your estimated house price is ready.")

        result_left, result_right = st.columns([1.5, 1])
        with result_left:
            st.markdown("### 💰 Estimated House Price")
            st.markdown(
                f"""
                <p style="
                    font-size: 2.5rem;
                    font-weight: 800;
                    color: #10b981;
                    margin: 0;
                ">
                    {format_inr(predicted_price)}
                </p>
                """,
                unsafe_allow_html=True,
            )

        with result_right:
            st.metric("🚗 Garage Spaces", garage)
            st.metric("📅 House Age", f"{age} years")

        detail_1, detail_2, detail_3, detail_4 = st.columns(4)
        with detail_1:
            st.metric("Price / sq ft", format_inr(predicted_price / area))
        with detail_2:
            st.metric("Total Rooms", bedrooms + bathrooms)
        with detail_3:
            st.metric("Year Built", 2026 - age)
        with detail_4:
            st.metric("Location Type", location)

    except ValueError as exc:
        st.error(f"⚠️ Invalid input: {exc}")
    except FileNotFoundError:
        st.error(
            "Model files not found. Train the model first:\n\n"
            "`python -m src.train`"
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        padding: 1rem 0 0.5rem 0;
    ">
        <strong>House Price Prediction Project</strong> &nbsp;|&nbsp;
        Built with Streamlit &bull; Scikit-learn &bull; Python<br>
        <span style="font-size: 0.8rem;">
        © 2026 ML Portfolio &nbsp;•&nbsp; Phase 16 – Deployment
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
