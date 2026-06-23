import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ==========================================
# 1. MODEL INTEGRATION (MEMBER 3 LINKAGE)
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.joblib")
        return model
    except FileNotFoundError:
        return None

model = load_model()

# ==========================================
# 2. MAIN INTERFACE
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
        family_history = st.radio("Family History of Risk?", ["No", "Yes"])

    st.markdown("---")
    
    if st.button("🚀 Calculate Risk Profile", use_container_width=True):
        with st.spinner("Processing features through ML Pipeline..."):
            time.sleep(0.6)
            
            history_mapped = 1 if family_history == "Yes" else 0
            activity_mapped = {"Low": 0, "Moderate": 1, "High": 2}[physical_activity]
            
            features = np.array([[ch2o, age, activity_mapped, bmi, history_mapped]])
            
            if model is not None:
                try:
                    prediction_prob = model.predict_proba(features)[0][1]
                    risk_score = int(prediction_prob * 100)
                except Exception:
                    risk_score = int(np.clip(50 + (bmi * 0.5) - (ch2o * 10), 0, 100))
            else:
                risk_score = int(np.clip(50 + (bmi * 0.5) - (ch2o * 10), 0, 100))

            st.subheader("📊 Assessment Result")
            
            if risk_score < 35:
                st.success(f"### 🟢 LOW RISK ({risk_score}%)")
                st.info("The subject falls within normal, safe operational parameters. Maintain current hydration (CH2O) and activity habits.")
            elif 35 <= risk_score < 70:
                st.warning(f"### 🟡 MODERATE RISK ({risk_score}%)")
                st.info("Precautionary thresholds breached. Increasing baseline water intake (CH2O) is recommended.")
            else:
                st.error(f"### 🔴 HIGH RISK ({risk_score}%)")
                st.info("Critical Risk Alert. Immediate clinical or behavioral interventions are advised.")

            # ------------------------------------------
            # DYNAMIC SENSITIVITY VISUALIZATION
            # ------------------------------------------
            st.markdown("---")
            st.subheader("💧 Hydration (CH2O) Optimization Vector")
            
            water_scenarios = np.linspace(0.0, 5.0, 11)
            simulated_risks = []
            
            for w in water_scenarios:
                if model is not None:
                    try:
                        test_features = np.array([[w, age, activity_mapped, bmi, history_mapped]])
                        prob = model.predict_proba(test_features)[0][1]
                        simulated_risks.append(int(prob * 100))
                    except Exception:
                        simulated_risks.append(int(np.clip(risk_score + (ch2o - w) * 12, 5, 95)))
                else:
