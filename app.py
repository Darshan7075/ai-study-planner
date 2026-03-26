import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from reportlab.pdfgen import canvas

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Study Planner", layout="wide")

# ------------------ CACHE ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip().str.lower()
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    return df.dropna()

@st.cache_resource
def train_model(df):
    X = df[[
        "sleep_hours",
        "focus_level",
        "stress_level",
        "screen_time",
        "physical_activity",
        "attendance",
        "previous_marks"
    ]]
    
    y = df["performance"]

    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# ------------------ CLUSTERING ------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(df[["study_hours", "performance"]])

# ------------------ PDF FUNCTION ------------------
def create_pdf(score, category):
    c = canvas.Canvas("study_report.pdf")
    c.drawString(100, 800, "AI Study Planner Report")
    c.drawString(100, 760, f"Predicted Score: {round(score,2)}%")
    c.drawString(100, 720, f"Category: {category}")
    c.drawString(100, 680, "Follow the recommended timetable!")
    c.save()

# ------------------ UI ------------------
st.title("🔥 AI-Based Smart Study Planner")

st.sidebar.header("🧾 Enter Your Details")

sleep = st.sidebar.slider("Sleep Hours", 4, 10, 7)
focus = st.sidebar.slider("Focus Level", 1, 10, 5)
stress = st.sidebar.slider("Stress Level", 1, 10, 5)
screen = st.sidebar.slider("Screen Time", 0, 10, 3)
physical = st.sidebar.slider("Physical Activity", 0, 5, 1)
attendance = st.sidebar.slider("Attendance", 50, 100, 80)
previous = st.sidebar.slider("Previous Marks", 0, 100, 70)

# ------------------ AI RESPONSE FUNCTION ------------------
def get_ai_response(text):
    text = text.lower()

    if "focus" in text:
        return "👉 Try Pomodoro technique (25 min study + 5 min break)"
    elif "stress" in text:
        return "👉 Take breaks, meditate, and stay calm"
    elif "time" in text:
        return "👉 Make a daily timetable and follow it consistently"
    elif "marks" in text:
        return "👉 Practice previous papers and revise regularly"
    else:
        return "👉 Stay consistent, revise daily, and avoid distractions"

# ------------------ MAIN BUTTON ------------------
if st.sidebar.button("🚀 Generate Plan"):

    user_data = [[sleep, focus, stress, screen, physical, attendance, previous]]
    predicted_score = model.predict(user_data)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 AI Prediction")
        st.success(f"🎯 Expected Score: {round(predicted_score,2)}%")

        if predicted_score > 85:
            category = "🏆 Top Performer"
        elif predicted_score > 60:
            category = "⚖️ Average Student"
        else:
            category = "⚠️ Needs Improvement"

        st.write(f"🧠 Category: {category}")

        st.subheader("📅 Study Advice")

        if predicted_score < 60:
            st.write("📌 Increase study hours and reduce distractions")
        elif predicted_score < 85:
            st.write("📌 Focus on revision and practice")
        else:
            st.write("🔥 Maintain consistency and practice mock tests")

    with col2:
        st.subheader("📊 Focus vs Performance")

        fig, ax = plt.subplots()
        ax.scatter(df["focus_level"], df["performance"], alpha=0.3)
        ax.scatter(focus, predicted_score, color='red', s=100)

        ax.set_xlabel("Focus Level")
        ax.set_ylabel("Performance")

        st.pyplot(fig)

    # ------------------ EXTRA ANALYSIS ------------------
    st.subheader("🧩 Student Group Analysis")

    cluster_result = kmeans.predict([[sleep, predicted_score]])[0]

    if cluster_result == 0:
        st.write("📘 Group 1: Average Learner")
    elif cluster_result == 1:
        st.write("🏆 Group 2: Top Performer")
    else:
        st.write("⚠️ Group 3: Needs Improvement")

    # ------------------ MODEL ACCURACY ------------------
    st.subheader("📈 Model Accuracy")

    y_pred = model.predict(df[[
        "sleep_hours",
        "focus_level",
        "stress_level",
        "screen_time",
        "physical_activity",
        "attendance",
        "previous_marks"
    ]])

    accuracy = r2_score(df["performance"], y_pred)
    st.info(f"Model Accuracy: {round(accuracy*100,2)}%")

    # ------------------ SMART RECOMMENDATIONS ------------------
    st.subheader("🤖 Smart Recommendations")

    if focus < 5:
        st.warning("📌 Improve focus using Pomodoro Technique")
    if stress > 7:
        st.warning("🧘 Reduce stress with meditation")
    if screen > 6:
        st.warning("📵 Reduce screen time")
    if physical < 1:
        st.warning("🏃 Increase physical activity")
    if predicted_score > 85:
        st.success("🔥 You're performing excellently!")

    # ------------------ TIMETABLE ------------------
    st.subheader("📅 Personalized Timetable")

    if predicted_score < 60:
        st.info("🕒 2–3 Hours Plan:\n• Basics Study\n• Practice\n• Revision")
    elif predicted_score < 85:
        st.success("🕒 4–5 Hours Plan:\n• Core Subjects\n• Practice\n• Revision")
    else:
        st.warning("🕒 6+ Hours Plan:\n• Deep Study\n• Practice\n• Revision")

    # ------------------ PDF ------------------
    st.subheader("📄 Download Report")

    create_pdf(predicted_score, category)

    with open("study_report.pdf", "rb") as file:
        st.download_button(
            label="📥 Download Your Report",
            data=file,
            file_name="study_report.pdf",
            mime="application/pdf"
        )

# ------------------ CHATBOT ------------------
st.subheader("🤖 AI Study Assistant")

user_q = st.text_input("Ask anything about study", key="chat_input")

if user_q:
    st.success(get_ai_response(user_q))

# ------------------ VOICE INPUT ------------------
st.subheader("🎤 Voice Input")

try:
    import speech_recognition as sr
except:
    st.warning("⚠️ speech_recognition not installed")

if st.button("🎤 Speak Now"):
    try:
        r = sr.Recognizer()

        with sr.Microphone() as source:
            st.info("Listening... बोलो अब 🎙️")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5, phrase_time_limit=7)

        text = r.recognize_google(audio)
        st.success(f"You said: {text}")

        st.write(get_ai_response(text))

    except Exception as e:
        st.error(f"❌ Voice error: {e}")