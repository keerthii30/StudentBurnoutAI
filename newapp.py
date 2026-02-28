import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Student Burnout AI System",
    layout="wide"
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ===================================================
# HOME PAGE
# ===================================================
if st.session_state.page == "home":

    st.title("AI-Based Student Burnout & Dropout Early Detection System")

    st.markdown("---")

    # -------------------------------
    # PROBLEM STATEMENT
    # -------------------------------
    st.header("Problem Statement")

    st.markdown("""
    Higher education institutions often detect student burnout and dropout risk 
    only after academic performance has significantly declined.

    Traditional monitoring approaches rely on:
    - End-semester grades
    - Manual attendance checks
    - Reactive faculty observations

    These methods delay intervention and reduce the effectiveness of support systems.

    There is a need for a data-driven early warning system capable of identifying 
    behavioral warning signals before academic failure occurs.
    """)

    st.markdown("---")

    # -------------------------------
    # OBJECTIVE
    # -------------------------------
    st.header("System Objective")

    st.markdown("""
    The objective of this AI system is to:

    • Detect early burnout indicators  
    • Estimate dropout probability  
    • Generate a continuous risk score (0–100)  
    • Explain behavioral factors influencing predictions  
    • Recommend targeted intervention strategies  

    This enables proactive and evidence-based academic decision-making.
    """)

    st.markdown("---")

    # -------------------------------
    # WORKFLOW
    # -------------------------------
    st.header("System Workflow")

    st.markdown("""
    1. Data Collection – LMS activity, attendance, submission behavior, sentiment metrics  
    2. Data Preprocessing – Encoding and scaling behavioral features  
    3. Behavioral Segmentation – KMeans clustering  
    4. Dropout Prediction – Logistic Regression  
    5. Burnout Risk Scoring – Random Forest Regression  
    6. Explainable AI – SHAP feature contribution analysis  
    7. Intervention Engine – Risk-based action recommendations  
    """)

    st.markdown("---")

    # -------------------------------
    # WHAT IT DOES
    # -------------------------------
    st.header("What This System Does")

    st.markdown("""
    ✔ Classifies students into behavioral risk segments  
    ✔ Predicts dropout probability  
    ✔ Calculates burnout risk score  
    ✔ Identifies key behavioral triggers  
    ✔ Provides interpretable AI explanations  
    ✔ Suggests intervention strategies  
    """)

    st.markdown("---")

    # -------------------------------
    # MODELS USED
    # -------------------------------
    st.header("AI Models Used")

    st.markdown("""
    • KMeans Clustering – Behavioral segmentation  
    • Logistic Regression – Dropout classification  
    • Random Forest Regression – Continuous risk scoring  
    • SHAP – Explainable AI layer  
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("Start Student Risk Assessment", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


# ===================================================
# DASHBOARD PAGE
# ===================================================
elif st.session_state.page == "dashboard":

    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("Student Risk Assessment Dashboard")

    # ---------------------------------------------------
    # LOAD MODELS
    # ---------------------------------------------------
    rf_model = joblib.load("rf_model.pkl")
    log_model = joblib.load("log_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # ---------------------------------------------------
    # SIDEBAR INPUTS
    # ---------------------------------------------------
    st.sidebar.header("Student Behavioral Inputs")

    lms = st.sidebar.slider("LMS Logins per Week", 0, 50, 20)
    login_trend = st.sidebar.selectbox("Login Trend", ["increase","stable","decrease"])
    delay = st.sidebar.slider("Avg Submission Delay (Days)", 0.0, 10.0, 2.0)
    missed = st.sidebar.slider("Missed Assignments", 0, 10, 1)
    attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 75.0)
    attendance_trend = st.sidebar.selectbox("Attendance Trend", ["increase","stable","decrease"])
    sentiment = st.sidebar.slider("Feedback Sentiment Score", -1.0, 1.0, 0.0)
    variance = st.sidebar.slider("Activity Variance", 0.0, 15.0, 5.0)
    late = st.sidebar.slider("Late Night Activity Ratio", 0.0, 1.0, 0.2)

    trend_map = {"increase":2, "stable":1, "decrease":0}

    input_data = pd.DataFrame([[
        lms,
        trend_map[login_trend],
        delay,
        missed,
        attendance,
        trend_map[attendance_trend],
        sentiment,
        variance,
        late
    ]], columns=[
        "lms_logins_per_week",
        "login_trend_change",
        "avg_submission_delay_days",
        "missed_assignments_count",
        "attendance_percent",
        "attendance_trend_change",
        "feedback_sentiment_score",
        "activity_variance",
        "late_night_activity_ratio"
    ])

    # ---------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------
    scaled_input = scaler.transform(input_data)

    cluster = kmeans.predict(scaled_input)[0]
    dropout_prob = log_model.predict_proba(input_data)[0][1]
    risk_score = float(np.clip(rf_model.predict(input_data)[0], 0, 100))

    # Risk category
    if risk_score < 33:
        category = "Low"
    elif risk_score < 66:
        category = "Medium"
    else:
        category = "High"

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{risk_score:.2f}/100")
    col2.metric("Risk Category", category)
    col3.metric("Dropout Probability", f"{dropout_prob*100:.2f}%")

    st.markdown("---")

    # ---------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # ---------------------------------------------------
    st.subheader("Global Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------------------------------------------
    # SHAP LOCAL EXPLANATION
    # ---------------------------------------------------
    st.subheader("Individual Risk Explanation (SHAP)")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(input_data)

    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact on Risk Score": shap_values.values[0]
    }).sort_values(by="Impact on Risk Score", key=abs, ascending=False)

    fig_shap = px.bar(
        shap_df,
        x="Impact on Risk Score",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig_shap, use_container_width=True)

    # ---------------------------------------------------
    # INTERVENTION ENGINE
    # ---------------------------------------------------
    st.subheader("Recommended Intervention Strategy")

    if risk_score < 33:
        st.success("""
        Maintain academic mentoring  
        Encourage leadership activities  
        Monthly engagement monitoring  
        """)
    elif risk_score < 66:
        st.warning("""
        Schedule mentoring session  
        Monitor assignment submissions  
        Conduct stress management workshop  
        """)
    else:
        st.error("""
        Immediate intervention required  
        Psychological counseling referral  
        Personalized recovery plan  
        Weekly attendance monitoring  
        """)

    st.markdown("---")
    st.caption("AI Behavioral Analytics System | KMeans + Logistic Regression + Random Forest + SHAP")