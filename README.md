# 🎓 AI-Based Student Burnout & Dropout Risk Prediction System

## 📌 Project Overview

This project develops a hybrid Machine Learning system to detect early signs of student burnout and dropout risk using behavioral and academic engagement indicators. 

The system analyzes engagement patterns such as LMS activity, attendance, assignment behavior, and sentiment trends to generate:

- 📊 Dropout Probability
- 📈 Burnout Risk Score (0–100)
- 🧠 Risk Category (Low / Medium / High)

The goal is to enable early intervention and proactive academic support.

---

## 🚀 Features

- Behavioral segmentation using KMeans Clustering
- Dropout probability prediction using Logistic Regression
- Continuous burnout risk scoring using Random Forest
- Feature scaling using StandardScaler
- Interactive Streamlit web dashboard
- Interpretable and explainable AI-driven insights

---

## 📂 Project Structure
StudentBurnoutAI/
│
├── app.py # Streamlit application
├── kmeans_model.pkl # KMeans clustering model
├── log_model.pkl # Logistic Regression model
├── rf_model.pkl # Random Forest model
├── scaler.pkl # Feature scaler
├── student_burnout_behavioral_dataset_200.csv
├── requirements.txt
├── .gitignore
└── README.md


---

## 📊 Dataset Information

### Dataset Type: Synthetic

### Why Synthetic?
There is no publicly available dataset that captures real student behavioral burnout signals such as LMS login trends, late-night activity, and submission delays. Therefore, a simulated dataset was generated to model realistic behavioral patterns.

---

## 📈 Dataset Generation Process

The dataset was created using statistical distributions and logical behavioral rules:

- LMS Logins → Normal distribution
- Attendance Percentage → Uniform distribution (60–100%)
- Submission Delay → Poisson distribution
- Missed Assignments → Random integer distribution
- Sentiment Score → Uniform distribution (-1 to 1)
- Activity Variance → Random variation model
- Late Night Activity Ratio → Random proportional distribution

Behavioral trends (increase/stable/decrease) were generated to simulate engagement shifts over time.

---

## 📌 Number of Records

- 200 synthetic student records

---

## 📑 Feature Description

| Feature | Description |
|----------|------------|
| lms_logins_per_week | Average LMS engagement frequency |
| login_trend_change | Engagement trend (increase/stable/decrease) |
| avg_submission_delay_days | Average delay in assignment submission |
| missed_assignments_count | Total missed assignments |
| attendance_percent | Overall attendance percentage |
| attendance_trend_change | Attendance trend pattern |
| feedback_sentiment_score | Sentiment score from feedback (-1 to 1) |
| activity_variance | Irregularity in study patterns |
| late_night_activity_ratio | Ratio of late-night study behavior |

---

## 🤖 Models Used

### 1️⃣ KMeans Clustering
- Segments students into behavioral groups

### 2️⃣ Logistic Regression
- Predicts dropout probability

### 3️⃣ Random Forest Regression
- Generates burnout risk score (0–100)

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- Confusion Matrix
- R² Score
- MAE
- RMSE

Random Forest showed superior performance in modeling non-linear behavioral relationships.

---

## 🧠 Behavioural Insights

- Declining LMS engagement is an early risk signal.
- Attendance below 70–75% increases dropout probability.
- Frequent assignment delays correlate with burnout.
- Negative feedback sentiment reflects disengagement.
- Risk increases when multiple weak signals appear together.

---


## 🎯 Practical Impact

This system enables:

- Early identification of at-risk students
- Data-driven academic intervention
- Improved student retention
- AI-supported institutional decision-making

---

## 📌 Future Scope

- Real-time LMS integration
- Deep learning time-series modeling
- Reinforcement learning-based intervention recommendation
- Institutional dashboard deployment

---

## 📄 License

This project is developed for academic and research purposes.
