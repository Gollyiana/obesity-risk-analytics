import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ==========================================
# 1. FLOATING REPAIR: RAW STRINGS FOR CSS
# ==========================================
# Wrapping this in a raw string (r""") prevents Streamlit Cloud from misinterpreting 
# the CSS curly brackets as Python string formatting parameters.
custom_css = r"""
<style>
    .reportview-container { background: #f0f2f6; }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 15px;
    }
    .low-risk { background-color: #2ecc71; }
    .mod-risk { background-color: #f39c12; }
    .high-risk { background-color: #e74c3c; }
</style>
"""
st.markdown(custom_css, unsafe_with_html=True)

# ==========================================
# 2. MODEL INTEGRATION (MEMBER 3 LINKAGE)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # Tries loading the project pickle/joblib file
        model = joblib.load("model.joblib")
        return model
    except FileNotFoundError:
        # Fallback dummy logic if Member 3's model isn't uploaded yet
        return None

model = load_model()

# ==========================================
# 3. PAGE INTERFACE & MULTI-TAB ARCHITECTURE
# ==========================================
st.title("⚖️ Predictive Risk Assessment & System Architecture")
st.caption("Integrated Full-Stack Web Application | Designed & Maintained by Aqif")

tab1, tab2, tab3 = st.tabs([
    "🎯 Predictive Risk Calculator", 
    "🏗️ System Architecture", 
    "📈 Performance & Conclusions"
])

# ------------------------------------------
# TAB 1: PREDICTIVE RISK CALCULATOR
# ------------------------------------------
with tab1:
    st.header("Patient Feature Input & Real-Time Risk Prediction")
    st.write("Adjust the features below to compute the instantaneous risk factor utilizing the backend ML pipeline.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💧 Behavioral Metrics")
        ch2o = st.slider("Daily Water Intake (CH2O in Liters)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        age = st.slider("Age", min_value=1, max_value=100, value=30)
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

    with col2:
        st.subheader("📋 Clinical Metrics")
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=24.5, step=0.1)
        family_history = st.radio("Family History of Risk?",
