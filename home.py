import streamlit as st
import pandas as pd

# -----------------------------
# Data URL
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/Gollyiana/obesity-risk-analytics/refs/heads/main/ObesityDataSet_raw_and_data_sinthetic.csv"


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


# -----------------------------
# Home Page Content
# -----------------------------
def home_page_content():
    st.header("Obesity Risk Analytics", divider="grey")

    st.write(
        """
        This application presents an obesity risk analytics dashboard using an obesity dataset.
        It allows users to explore obesity patterns based on demographic, lifestyle, and health-related factors.
        """
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(label="PLO 2", value="3.3", help="PLO 2: Cognitive Skill")
    col2.metric(label="PLO 3", value="3.5", help="PLO 3: Digital Skill")
    col3.metric(label="PLO 4", value="4.0", help="PLO 4: Interpersonal Skill")
    col4.metric(label="PLO 5", value="4.3", help="PLO 5: Communication Skill")

    st.divider()

    df2 = load_data()

    # Dataset preview
    st.subheader("Dataset Preview")
    st.write(f"Dataset contains **{df2.shape[0]} rows** and **{df2.shape[1]} columns**.")
    st.dataframe(df2.head(), use_container_width=True)

    st.info("Use the sidebar to open the Insights Dashboard page.")


# -----------------------------
# Custom Navigation Names
# -----------------------------
home_page = st.Page(
    home_page_content,
    title="Obesity Risk Analytics",
    icon="🏥",
    default=True
)

# Added your tab right here as the first entry under home!
golinia_page = st.Page(
    "pages/tab_1_golinia.py",
    title="Cohort Data Diagnostics",
    icon="📋"
)

insights_page = st.Page(
    "pages/tab_2_hidayah.py",
    title="Insights Dashboard",
    icon="📊"
)

model_page = st.Page(
    "pages/tab_3_arissa.py",
    title="Model Evaluation Hub",
    icon="🤖"
)

model_page = st.Page(
    "pages/tab_4_aqif.py",
    title="Risk Analytics Dashboard",
    icon="⚖️"
)

# -----------------------------
# Navigation
# -----------------------------
# Included golinia_page into the active navigation array layout
pg = st.navigation([home_page, golinia_page, insights_page, model_page, risk_page])
pg.run()
