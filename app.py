import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Study Planner", layout="wide")

# ------------------ PREMIUM UI ------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}
.stButton>button {
    background: linear-gradient(45deg, #3b82f6, #06b6d4);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Study Planner Dashboard")

# ------------------ LOAD MODELS ------------------
rg = joblib.load("student_performance_model.pkl")
clf_model = joblib.load("burnout_model.pkl")
le = joblib.load("label_encoder.pkl")

# ------------------ USER DETAILS ------------------
st.subheader("👤 Student Details")

col1, col2, col3 = st.columns(3)
name = col1.text_input("Name")
enrollment = col2.text_input("Enrollment No")
college = col3.text_input("College Name")

st.divider()

# ------------------ INPUT SECTION ------------------
st.subheader("📊 Study Inputs")

c1, c2, c3 = st.columns(3)
study_hours = c1.slider("Study Hours", 0.0, 12.0, 4.0)
sleep_hours = c2.slider("Sleep Hours", 0.0, 12.0, 7.0)
focus_level = c3.slider("Focus Level", 1, 10, 5)

c4, c5, c6 = st.columns(3)
stress_level = c4.slider("Stress Level", 1, 10, 5)
break_time = c5.slider("Break Time (min)", 0, 120, 30)
revision = c6.selectbox("Revision", [0,1])

c7, c8, c9 = st.columns(3)
screen_time = c7.slider("Screen Time", 0.0, 12.0, 4.0)
physical_activity = c8.selectbox("Physical Activity", [0,1])
attendance = c9.slider("Attendance %", 0, 100, 75)

previous_marks = st.slider("Previous Marks", 0, 100, 60)

st.divider()

# ------------------ FUNCTIONS ------------------

def recommend(row):
    tips = []

    # Condition-based tips
    if row['study_hours'] < 3:
        tips.append("Increase study time (3–4 hrs daily).")
    elif row['study_hours'] > 8:
        tips.append("You're studying a lot. Take breaks to avoid burnout.")

    if row['sleep_hours'] < 6:
        tips.append("Sleep at least 7–8 hrs.")

    if row['stress_level'] > 7:
        tips.append("Reduce stress (meditation recommended).")

    if row['screen_time'] > 6:
        tips.append("Reduce screen time.")

    if row['focus_level'] < 5:
        tips.append("Use Pomodoro technique.")

    if row['physical_activity'] == 0:
        tips.append("Add daily exercise.")

    if row['revision'] == 0:
        tips.append("Start regular revision.")

    if row['burnout_risk'] == 'High Risk':
        tips.append("High burnout risk — take proper rest!")

    # Default fallback tips
    default_tips = [
        "Stay consistent with your daily study routine.",
        "Revise topics regularly to improve memory.",
        "Avoid distractions while studying.",
        "Maintain a balanced study-life schedule."
    ]

    i = 0
    while len(tips) < 4 and i < len(default_tips):
        if default_tips[i] not in tips:
            tips.append(default_tips[i])
        i += 1

    return tips[:4]


def generate_roadmap(row):
    study_hours = row['study_hours']
    sleep_hours = row['sleep_hours']
    previous_marks = row['previous_marks']

    roadmap = []

    roadmap.append("📅 WEEK 1 — Foundation & Habit Building")
    roadmap.append("• Set daily study routine")
    roadmap.append("• Fix sleep schedule")

    roadmap.append("\n📅 WEEK 2 — Revision & Focus")
    roadmap.append("• Add 30 min revision daily")
    roadmap.append("• Reduce distractions")

    roadmap.append("\n📅 WEEK 3 — Practice")
    roadmap.append("• Solve test papers")
    roadmap.append("• Improve weak areas")

    roadmap.append("\n📅 WEEK 4 — Final Preparation")
    roadmap.append("• Revise all topics")
    roadmap.append("• Focus on important subjects")

    roadmap.append("\n💡 Tip: Consistency > Intensity")

    return roadmap

# ------------------ BUTTON ------------------

if st.button("🚀 Predict Performance"):

    student = {
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "focus_level": focus_level,
        "stress_level": stress_level,
        "break_time": break_time,
        "revision": revision,
        "screen_time": screen_time,
        "physical_activity": physical_activity,
        "attendance": attendance,
        "previous_marks": previous_marks
    }

    df = pd.DataFrame([student])

    df['effective_time'] = df['study_hours'] - (df['break_time']/60)
    df['sleep_quality'] = df['sleep_hours']

    features = ['study_hours', 'sleep_hours', 'focus_level', 'stress_level',
                'break_time', 'revision', 'screen_time', 'physical_activity',
                'attendance', 'previous_marks', 'effective_time', 'sleep_quality']

    # Prediction
    score = rg.predict(df[features])[0]

    burnout_pred = clf_model.predict(df[features])[0]
    burnout = le.inverse_transform([burnout_pred])[0]

    df['burnout_risk'] = burnout

    tips = recommend(df.iloc[0])
    roadmap = generate_roadmap(df.iloc[0])

    # ------------------ OUTPUT ------------------
    st.divider()
    st.subheader("📊 Results")

    st.markdown(f"""
    <div class="card">
    👤 <b>Name:</b> {name} <br>
    🆔 <b>Enrollment:</b> {enrollment} <br>
    🏫 <b>College:</b> {college}
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Score", f"{score:.2f}%")
    col2.metric("Performance", "High" if score>=80 else "Medium" if score>=60 else "Low")
    col3.metric("Burnout Risk", burnout)

    st.divider()

    st.subheader("💡 Smart Suggestions")
    for tip in tips:
        st.markdown(f"<div class='card'>👉 {tip}</div>", unsafe_allow_html=True)

    st.subheader("🗺️ Study Roadmap")
    for r in roadmap:
        st.markdown(f"<div class='card'>{r}</div>", unsafe_allow_html=True)