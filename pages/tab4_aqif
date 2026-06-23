import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Data URL (Consistent with Team Setup)
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/Gollyiana/obesity-risk-analytics/refs/heads/main/ObesityDataSet_raw_and_data_sinthetic.csv"


# -----------------------------
# Load Dataset with Cache
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


df_raw = load_data()

# -----------------------------
# Data Processing Pipeline (Member 1 Task)
# -----------------------------
df_cleaned = df_raw.copy()

# Rounding floating-point anomalies caused by SMOTE synthetic upsampling
df_cleaned["Age"] = df_cleaned["Age"].round(1)
df_cleaned["Weight"] = df_cleaned["Weight"].round(1)
df_cleaned["Height"] = df_cleaned["Height"].round(2)
df_cleaned["FCVC"] = df_cleaned["FCVC"].round(0).astype(int)  # Veg consumption
df_cleaned["NCP"] = df_cleaned["NCP"].round(0).astype(int)    # Main meals
df_cleaned["CH2O"] = df_cleaned["CH2O"].round(1)              # Water intake
df_cleaned["FAF"] = df_cleaned["FAF"].round(1)                # Physical activity
df_cleaned["TUE"] = df_cleaned["TUE"].round(1)                # Technology use

# -----------------------------
# Page Title & Layout
# -----------------------------
st.title("📊 Cohort Profile & Data Diagnostics")
st.write(
    "Welcome to the **Data Diagnostics and Overview Hub**. "
    "This section details the initial data ingestion, resolution of synthetic formatting quirks, "
    "and structural profile characteristics of the study cohort."
)

st.divider()

# -----------------------------
# Key Metrics
# -----------------------------
st.subheader("📌 Baseline Cohort Indicators")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records Ingested", df_cleaned.shape[0])
col2.metric("Cohort Mean Age", f"{df_cleaned['Age'].mean():.1f} Years")
col3.metric("Cohort Mean Weight", f"{df_cleaned['Weight'].mean():.1f} kg")
col4.metric("Family History Rate", f"{(df_cleaned['family_history_with_overweight'] == 'yes').mean() * 100:.1f}%")

st.divider()

# -----------------------------
# Demonstration of Data Cleaning (Quirk Rectification)
# -----------------------------
st.subheader("🛠️ Data Engineering: Synthetic Quirk Rectification")
st.write(
    "Because the dataset includes synthetic records generated via SMOTE oversampling, "
    "discrete attributes like the number of main meals (`NCP`) or frequency of vegetable consumption (`FCVC`) "
    "originally contained unrealistic fractional values. Below is a comparison of the raw vs. engineered attributes:"
)

view_mode = st.radio("Toggle Processing View:", ["Show Cleaned Data (Recommended)", "Show Raw Unprocessed Data"])
display_df = df_cleaned if view_mode == "Show Cleaned Data (Recommended)" else df_raw

st.dataframe(display_df[["Age", "Gender", "Height", "Weight", "FCVC", "NCP", "CH2O"]].head(10), use_container_width=True)

st.divider()

# -----------------------------
# Statistical Summary Matrix
# -----------------------------
st.subheader("📋 Numerical Descriptive Summary Matrix")
st.write(
    "This summary matrix displays central tendency metrics, standard deviations, and range limits "
    "across all numeric attributes in the study cohort."
)

st.dataframe(df_cleaned.describe().T, use_container_width=True)

st.divider()

# -----------------------------
# Interactive Data Record Explorer
# -----------------------------
st.subheader("🔍 Interactive Cohort Explorer")
st.write("Isolate and examine rows by choosing a slice of data to observe below.")

row_slider = st.slider("Select slice size to observe:", min_value=5, max_value=100, value=15)
st.dataframe(df_cleaned.head(row_slider), use_container_width=True)

st.divider()

# -----------------------------
# Report/Contextual Insight Summary
# -----------------------------
st.subheader("💡 Behavioral Assessment Rationale")
st.info(
    "**Domain Analyst Note:** Traditional diagnostics categorize patient wellness status relying purely "
    "on Body Mass Index measurements. This framework addresses the behavioral catalysts behind those metrics, "
    "allowing healthcare providers to examine active transportation methods, technology consumption hours, "
    "and hydration volumes to target root lifestyle habits."
)
