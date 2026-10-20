
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Virtual Climate Modeling",
    page_icon="🌍",
    layout="wide",
)

# Colourful Climate-Tech Theme


# Editorial Climate-Tech Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

:root {
    --cream: #F7F1E7;
    --paper: #FFFCF7;
    --ink: #302B25;
    --muted: #81776B;
    --terracotta: #C65D3B;
    --burnt-orange: #E68A46;
    --sage: #71835B;
    --pale-sage: #E6EAD9;
    --peach: #F5D5C2;
    --gold: #D9AD57;
    --border: #E7DACA;
}

html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Poppins', sans-serif !important;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: var(--cream) !important;
    color: var(--ink) !important;
}

.block-container {
    max-width: 1500px;
    padding: 2rem 2.2rem 3rem;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: -0.6px;
}

h1 {
    color: var(--ink) !important;
    font-size: 2.45rem !important;
    font-weight: 800 !important;
}

h2 {
    color: var(--ink) !important;
    font-size: 1.65rem !important;
    font-weight: 750 !important;
}

h3 {
    color: var(--terracotta) !important;
    font-weight: 700 !important;
}

p, li, label {
    font-family: 'Poppins', sans-serif !important;
    color: var(--ink);
}

[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #EAE1D2 !important;
    border-right: 1px solid #DCCDBA;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

section[data-testid="stSidebar"] * {
    font-family: 'Poppins', sans-serif !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: var(--ink) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 11px 13px;
    margin: 5px 0;
    border-radius: 12px;
    border: 1px solid transparent;
    background: rgba(255, 252, 247, 0.65);
    transition: all .2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #FFFCF7;
    border-color: #D8A48A;
}

/* Hero banner */
.climate-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(118deg, #733D2C 0%, #C65D3B 52%, #E68A46 100%);
    padding: 38px 36px;
    border-radius: 24px;
    margin: 0 0 28px 0;
    box-shadow: 0 14px 32px rgba(123, 65, 43, .16);
}

.climate-hero:after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    right: -60px;
    top: -100px;
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 50%;
    box-shadow:
        0 0 0 25px rgba(255,255,255,.05),
        0 0 0 55px rgba(255,255,255,.04);
}

.hero-eyebrow {
    color: #FFE7C9 !important;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: 2.3px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    color: #FFFFFF !important;
    font-size: clamp(2rem, 4vw, 3.25rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -1.5px;
    margin: 0 0 15px 0;
    max-width: 760px;
}

.hero-description {
    color: #FFF1E5 !important;
    font-size: 1rem;
    line-height: 1.8;
    max-width: 700px;
    margin: 0;
}

.hero-tag {
    display: inline-block;
    margin-top: 22px;
    padding: 8px 13px;
    border: 1px solid rgba(255,255,255,.4);
    background: rgba(255,255,255,.12);
    border-radius: 999px;
    color: white !important;
    font-size: .78rem;
    font-weight: 600;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px 20px;
    box-shadow: 0 6px 20px rgba(69, 48, 30, .055);
    transition: transform .2s ease, box-shadow .2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 25px rgba(69, 48, 30, .10);
}

div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: var(--terracotta) !important;
    font-weight: 800 !important;
}

/* Tables */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 15px;
    overflow: hidden;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 13px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: var(--terracotta) !important;
    color: white !important;
    border: 1px solid var(--terracotta) !important;
    border-radius: 10px !important;
    padding: .6rem 1rem;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #A84B2F !important;
    border-color: #A84B2F !important;
    transform: translateY(-2px);
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid #D8C8B6 !important;
    border-radius: 9px !important;
}

/* Links and dividers */
a {
    color: var(--terracotta) !important;
}

hr {
    border-color: var(--border) !important;
}

/* Images */
div[data-testid="stImage"] img {
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 4px;
    background: var(--paper);
}

/* Responsive layout */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .climate-hero {
        padding: 26px 22px;
        border-radius: 18px;
    }

    .hero-title {
        font-size: 2rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="climate-hero">
    <div class="hero-eyebrow">Climate research · Machine learning · Explainable AI</div>
    <div class="hero-title">Understanding our planet,<br>one data point at a time.</div>
    <p class="hero-description">
        Explore global temperature anomalies, evaluate predictive models,
        uncover model behaviour, and examine illustrative future trends
        through one interactive climate analytics workspace.
    </p>
    <div class="hero-tag">🌍 VIRTUAL CLIMATE MODELING &nbsp; · &nbsp; 1880–2025</div>
</div>
""", unsafe_allow_html=True)


st.info(
    "This project uses NASA GISTEMP temperature anomaly data. "
    "Future projections are simple extrapolations, not official climate forecasts."
)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# Sidebar navigation
section = st.sidebar.radio(
    "Explore the project",
    [
        "Overview",
        "Historical Temperature",
        "Model Comparison",
        "Uncertainty Analysis",
        "Explainable AI (SHAP)",
        "Future Projection",
        "Project Files",
    ],
)

# Overview
if section == "Overview":
    st.header("Project Overview")
    st.write(
        """
        This project explores historical global temperature anomalies and
        evaluates whether simple machine-learning models can predict annual
        changes in the series.

        It compares a previous-year baseline, Linear Regression, and Random
        Forest. It also examines model disagreement, feature importance,
        and an illustrative future trend.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Historical data", "1880–2025")
    col2.metric("ML models", "2")
    col3.metric("Evaluation", "Time-series split")

    st.subheader("Research questions")
    st.markdown(
        """
        - How has global temperature anomaly changed over time?
        - How do machine-learning models compare with a simple baseline?
        - Where do model predictions disagree?
        - Which features influence the fitted model?
        - What does a simple trend extrapolation suggest?
        """
    )

# Historical data
elif section == "Historical Temperature":
    st.header("Historical Global Temperature Anomalies")

    data_path = DATA_DIR / "global_temperature.csv"
    if data_path.exists():
        df = load_csv(data_path)
        st.write("Preview of the source dataset:")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(
            "The source file contains monthly anomaly values. "
            "The table above shows the original dataset preview."
        )
    else:
        st.warning("Source dataset not found in the data folder.")

    chart_path = OUTPUT_DIR / "global_temperature_trend.png"
    if chart_path.exists():
        st.image(
            str(chart_path),
            caption="Historical global temperature trend from project analysis",
            use_container_width=True,
        )
    else:
        st.warning("Historical trend chart not found in outputs.")

# Model comparison
elif section == "Model Comparison":
    st.header("Model Performance Comparison")
    path = OUTPUT_DIR / "model_comparison.csv"

    if path.exists():
        results = load_csv(path)
        st.dataframe(results, use_container_width=True)
        st.write("Lower MAE and RMSE generally indicate smaller prediction errors.")

        chart_path = OUTPUT_DIR / "model_predictions.png"
        if chart_path.exists():
            st.image(
                str(chart_path),
                caption="Observed and predicted values",
                use_container_width=True,
            )

        st.warning(
            "In the current experiment, the previous-year baseline performed "
            "better on the held-out test period than the tested ML models. "
            "The results do not establish that ML is superior."
        )
    else:
        st.error("Model comparison results were not found.")

# Uncertainty
elif section == "Uncertainty Analysis":
    st.header("Uncertainty and Model Disagreement")
    path = OUTPUT_DIR / "uncertainty_analysis.csv"

    if path.exists():
        df = load_csv(path)
        st.dataframe(df, use_container_width=True)

        st.caption(
            "The disagreement measure compares model predictions. "
            "It is not a calibrated prediction interval or a complete measure "
            "of climate uncertainty."
        )
    else:
        st.error("Uncertainty analysis results were not found.")

# Explainable AI
elif section == "Explainable AI (SHAP)":
    st.header("Explainable AI — Feature Importance")
    path = OUTPUT_DIR / "shap_feature_importance.csv"

    if path.exists():
        df = load_csv(path)
        st.dataframe(df, use_container_width=True)

        chart_path = OUTPUT_DIR / "shap_feature_importance.png"
        if chart_path.exists():
            st.image(
                str(chart_path),
                caption="SHAP feature-importance analysis",
                use_container_width=True,
            )

        st.caption(
            "Feature importance describes the fitted model's behavior. "
            "It does not prove that a feature causes global warming."
        )
    else:
        st.error("SHAP results were not found.")

# Projection
elif section == "Future Projection":
    st.header("Illustrative Future Projection")
    path = OUTPUT_DIR / "future_projection.csv"

    if path.exists():
        df = load_csv(path)
        st.dataframe(df, use_container_width=True)

        chart_path = OUTPUT_DIR / "future_projection.png"
        if chart_path.exists():
            st.image(
                str(chart_path),
                caption="Simple linear-trend extrapolation",
                use_container_width=True,
            )

        st.warning(
            "This is an illustrative statistical extrapolation, not an official "
            "climate forecast. It does not model emissions scenarios, ocean "
            "dynamics, or natural climate variability."
        )
    else:
        st.error("Future projection results were not found.")

# Files
elif section == "Project Files":
    st.header("Project Outputs")
    st.write("Files generated by the analysis:")

    if OUTPUT_DIR.exists():
        files = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_file())
        if files:
            for file in files:
                st.write(f"📄 {file.name}")
        else:
            st.info("No output files found.")
    else:
        st.warning("The outputs folder was not found.")

st.divider()
st.caption(
    "Virtual Climate Modeling | Educational research project | "
    "Interpret results with scientific caution"
)
