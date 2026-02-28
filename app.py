import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Student Burnout AI System",
    layout="wide",
    page_icon="🎓"
)

st.title("🎓 AI-Based Student Burnout & Dropout Analytics System")
st.markdown("Behavioral clustering + Risk scoring + Explainable AI Dashboard")

# -----------------------------
# LOAD MODELS
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
log_model = joblib.load("log_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# SIDEBAR INPUTS (SLICERS)
# -----------------------------
st.sidebar.header("📊 Student Behavioral Inputs")

lms = st.sidebar.slider("LMS Logins per Week", 0, 50, 20)
login_trend = st.sidebar.selectbox("Login Trend", ["increase","stable","decrease"])
delay = st.sidebar.slider("Avg Submission Delay (Days)", 0.0, 10.0, 2.0)
missed = st.sidebar.slider("Missed Assignments", 0, 10, 1)
attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 75.0)
attendance_trend = st.sidebar.selectbox("Attendance Trend", ["increase","stable","decrease"])
sentiment = st.sidebar.slider("Feedback Sentiment Score", -1.0, 1.0, 0.0)
variance = st.sidebar.slider("Activity Variance", 0.0, 15.0, 5.0)
late = st.sidebar.slider("Late Night Activity Ratio", 0.0, 1.0, 0.2)

# -----------------------------
# ENCODING
# -----------------------------
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

# -----------------------------
# MODEL PREDICTIONS
# -----------------------------

# Scale for clustering
scaled_input = scaler.transform(input_data)

# KMeans
cluster = kmeans.predict(scaled_input)[0]
cluster_map = {0:"🟢 Low Risk Segment", 1:"🟡 Medium Risk Segment", 2:"🔴 High Risk Segment"}
segment = cluster_map.get(cluster, "Unknown")

# Logistic Regression
dropout_prob = log_model.predict_proba(input_data)[0][1]

# Random Forest Risk Score
risk_score = rf_model.predict(input_data)[0]
risk_score = float(np.clip(risk_score, 0, 100))

# Risk Category
if risk_score < 33:
    category = "🟢 Low"
elif risk_score < 66:
    category = "🟡 Medium"
else:
    category = "🔴 High"

# -----------------------------
# DISPLAY METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📊 Risk Score", f"{risk_score:.2f}/100")
col2.metric("⚠ Risk Category", category)
col3.metric("🎯 Dropout Probability", f"{dropout_prob*100:.2f}%")

st.markdown("---")

# -----------------------------
# BEHAVIOR SEGMENT
# -----------------------------
st.subheader("🧠 Behavioral Cluster Segment")
st.info(segment)

# -----------------------------
# FEATURE IMPORTANCE (GLOBAL)
# -----------------------------
st.subheader("📌 Key Behavioral Triggers (Global Importance)")

importance_df = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=True)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Random Forest Feature Importance"
)

st.plotly_chart(fig_importance, use_container_width=True)

# -----------------------------
# SHAP EXPLAINABILITY (LOCAL)
# -----------------------------
st.subheader("🔍 Explainable AI – Individual Risk Breakdown")

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
    orientation="h",
    title="SHAP Feature Contribution (Local Explanation)"
)

st.plotly_chart(fig_shap, use_container_width=True)

# -----------------------------
# RISK GAUGE
# -----------------------------
st.subheader("📈 Risk Gauge")

gauge_fig = px.bar(
    x=["Risk Score"],
    y=[risk_score],
    text=[f"{risk_score:.2f}"],
)

gauge_fig.update_layout(
    yaxis=dict(range=[0,100]),
    title="Risk Score Scale (0-100)"
)

st.plotly_chart(gauge_fig, use_container_width=True)

# -----------------------------
# INTERVENTION RECOMMENDATION
# -----------------------------
st.subheader("💡 Recommended Intervention Strategy")

if risk_score < 33:
    st.success("""
    ✔ Maintain academic mentoring  
    ✔ Encourage leadership activities  
    ✔ Monitor engagement trends monthly  
    """)

elif risk_score < 66:
    st.warning("""
    ⚠ Schedule faculty mentoring session  
    ⚠ Monitor assignment submission closely  
    ⚠ Encourage peer group engagement  
    ⚠ Conduct stress management workshop  
    """)

else:
    st.error("""
    🚨 Immediate intervention required  
    🚨 Psychological counseling referral  
    🚨 Personalized academic recovery plan  
    🚨 Weekly attendance monitoring  
    🚨 Reduce workload temporarily  
    """)

st.markdown("---")
st.caption("AI Behavioral Analytics System | Built with KMeans + Logistic Regression + Random Forest + SHAP + Streamlit")