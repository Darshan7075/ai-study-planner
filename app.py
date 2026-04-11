import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import datetime
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Smart AI Study Planner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ SESSION STATE ------------------
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'user_name': "Guest Student",
        'user_id': "ST-1234",
        'study_hours': 4.0,
        'sleep_hours': 7.0,
        'focus_level': 6,
        'stress_level': 4,
        'screen_time': 3.0,
        'physical_activity': 0,
        'attendance': 85,
        'previous_marks': 75
    }

if 'results' not in st.session_state:
    st.session_state.results = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ------------------ PREMIUM UI ------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }
    
    .main-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        padding: 10px;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(59, 130, 246, 0.4);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .metric-card {
        text-align: center;
        padding: 25px;
        background: rgba(59, 130, 246, 0.05);
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        height: 100%;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    [data-testid="stHorizontalBlock"] {
        background: rgba(255, 255, 255, 0.02);
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    
    .nav-button {
        border-bottom: 2px solid transparent !important;
    }
    
    .nav-active {
        border-bottom: 2px solid #3b82f6 !important;
        background: rgba(59, 130, 246, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ CACHING & DATA LOADING ------------------
@st.cache_resource
def load_models():
    if not os.path.exists("student_performance_model.pkl"):
        try:
            # Check if training script exists, otherwise errors will handled by streamlit
            if os.path.exists("train_model.py"):
                from train_model import train_and_save
                train_and_save()
            else:
                st.error("⚠️ Training script 'train_model.py' not found.")
                return None, None, None, None
        except Exception as e:
            st.error(f"⚠️ Error during training: {e}")
            return None, None, None, None
            
    try:
        model = joblib.load("student_performance_model.pkl")
        kmeans = joblib.load("student_clusters.pkl")
        scaler = joblib.load("scaler.pkl")
        accuracy = joblib.load("model_accuracy.pkl")
        return model, kmeans, scaler, accuracy
    except:
        st.error("⚠️ Error loading models.")
        return None, None, None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data.csv")
    except:
        return pd.DataFrame()

model, kmeans, scaler, accuracy = load_models()
df_full = load_data()

# ------------------ CORE LOGIC ------------------
def classify_student(score):
    if score >= 85: return "Top Performer", "🔥"
    if score >= 65: return "Average Performer", "📈"
    return "Needs Improvement", "⚠️"

def get_recommendations(score, data):
    recs = []
    if score < 70:
        recs.append("Increase daily study hours to at least 6 hours.")
        recs.append("Reduce screen time and social media usage.")
    else:
        recs.append("Maintain consistency and start focusing on complex problem-solving.")
        
    if data['sleep_hours'] < 7:
        recs.append("Prioritize 7-8 hours of sleep for better cognitive function.")
    if data['focus_level'] < 6:
        recs.append("Practice meditation or the Pomodoro technique to improve focus.")
    if data['stress_level'] > 6:
        recs.append("Include more breaks and physical activity to manage stress.")
    
    return recs

def generate_timetable(score):
    schedule = {
        "Morning (08:00 - 11:00)": "Hard Subject / Problem Solving",
        "Afternoon (13:00 - 16:00)": "Practical / Lab Work / Calculations",
        "Evening (17:00 - 19:00)": "Revision & Notes Preparation",
        "Night (20:00 - 22:00)": "Light Reading / Planning for tomorrow"
    }
    if score < 60:
        schedule["Special Slot"] = "Doubt clearing with mentors (Extra 1 hour)"
    elif score > 90:
        schedule["Extra"] = "Advanced Project Work / Mentoring others"
    return schedule

def create_pdf(name, score, category, recs, user_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(190, 10, text="Smart AI Study Planner - Report", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(190, 10, text=f"Student Name: {name}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(190, 10, text=f"Student ID: {user_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(190, 10, text=f"Predicted Performance: {score:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(190, 10, text=f"Category: {category}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(190, 10, text="Key Recommendations:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for r in recs:
        pdf.multi_cell(180, 10, text=f"- {r}")
    pdf.ln(5)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(190, 10, text=f"Report Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

# ------------------ NAVIGATION ------------------
def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# Top Horizontal Navigation
st.markdown("<br>", unsafe_allow_html=True)
nav_cols = st.columns(6)
pages = ["Home", "Dashboard", "Study Planner", "Analytics", "AI Assistant", "Report"]
icons = ["🏠", "📊", "📅", "📈", "🤖", "📝"]

for i, (page, icon) in enumerate(zip(pages, icons)):
    is_active = st.session_state.page == page
    # Styling for active page
    btn_type = "primary" if is_active else "secondary"
    if nav_cols[i].button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True, type=btn_type):
        set_page(page)

st.divider()

# ------------------ PAGES ------------------

if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align: center; color: white;'>🚀 Smart AI Study Planner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #94a3b8;'>Enter your metrics to generate a personalized learning roadmap.</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Profile Information")
            st.session_state.form_data['user_name'] = st.text_input("Full Name", st.session_state.form_data['user_name'])
            st.session_state.form_data['user_id'] = st.text_input("Student ID", st.session_state.form_data['user_id'])
            
            st.subheader("📊 Personal Metrics")
            st.session_state.form_data['study_hours'] = st.slider("Daily Study Hours", 1.0, 12.0, float(st.session_state.form_data['study_hours']), 0.5)
            st.session_state.form_data['sleep_hours'] = st.slider("Daily Sleep Hours", 4.0, 10.0, float(st.session_state.form_data['sleep_hours']), 0.5)
            st.session_state.form_data['focus_level'] = st.slider("Focus Level (1-10)", 1, 10, int(st.session_state.form_data['focus_level']))
            st.session_state.form_data['stress_level'] = st.slider("Stress Level (1-10)", 1, 10, int(st.session_state.form_data['stress_level']))
        
        with col2:
            st.subheader("🎓 Academic Record")
            st.session_state.form_data['screen_time'] = st.slider("Screen Time (Hours)", 1.0, 10.0, float(st.session_state.form_data['screen_time']), 0.5)
            st.session_state.form_data['attendance'] = st.slider("Attendance %", 50, 100, int(st.session_state.form_data['attendance']))
            st.session_state.form_data['previous_marks'] = st.slider("Previous Marks %", 40, 100, int(st.session_state.form_data['previous_marks']))
            st.session_state.form_data['physical_activity'] = st.selectbox("Regular Physical Activity?", [0, 1], index=st.session_state.form_data['physical_activity'], format_func=lambda x: "Yes" if x==1 else "No")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Generate My AI Planner", use_container_width=True):
            if model is not None:
                # Prepare input
                input_df = pd.DataFrame([{
                    'study_hours': st.session_state.form_data['study_hours'],
                    'sleep_hours': st.session_state.form_data['sleep_hours'],
                    'focus_level': st.session_state.form_data['focus_level'],
                    'stress_level': st.session_state.form_data['stress_level'],
                    'screen_time': st.session_state.form_data['screen_time'],
                    'physical_activity': st.session_state.form_data['physical_activity'],
                    'attendance': st.session_state.form_data['attendance'],
                    'previous_marks': st.session_state.form_data['previous_marks']
                }])
                
                # Predict
                pred_score = model.predict(input_df)[0]
                category, icon = classify_student(pred_score)
                user_scaled = scaler.transform(input_df)
                cluster = kmeans.predict(user_scaled)[0]
                
                # Store results
                st.session_state.results = {
                    'pred_score': pred_score,
                    'category': category,
                    'icon': icon,
                    'cluster': cluster,
                    'recs': get_recommendations(pred_score, st.session_state.form_data),
                    'input_df': input_df
                }
                set_page("Dashboard")
            else:
                st.error("Error: ML Models not loaded.")
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "Dashboard":
    if st.session_state.results is None:
        st.warning("Please generate a plan first on the Home page.")
        if st.button("Go to Home"): set_page("Home")
    else:
        res = st.session_state.results
        st.title(f"📊 {st.session_state.form_data['user_name']}'s Dashboard")
        
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <h3>Predicted Score</h3>
                <h1 style='color: #3b82f6;'>{res['pred_score']:.1f}%</h1>
                <p>{res['icon']} {res['category']}</p>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <h3>Focus Level</h3>
                <h1 style='color: #10b981;'>{st.session_state.form_data['focus_level']*10}%</h1>
                <p>Efficiency Score</p>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <h3>Student Cluster</h3>
                <h1 style='color: #f59e0b;'>Group {res['cluster']}</h1>
                <p>Peer Comparison Group</p>
            </div>""", unsafe_allow_html=True)
        with m4:
            acc_val = (accuracy * 100) if accuracy else 0
            st.markdown(f"""<div class="metric-card">
                <h3>Model Accuracy</h3>
                <h1 style='color: #ef4444;'>{acc_val:.1f}%</h1>
                <p>Confidence Level</p>
            </div>""", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💡 Smart AI Recommendations")
            for r in res['recs']:
                st.markdown(f"<div style='padding: 10px; border-left: 3px solid #3b82f6; background: rgba(59, 130, 246, 0.05); margin-bottom: 10px;'>{r}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎯 Key Insight")
            st.info(f"The AI predicts you will achieve **{res['pred_score']:.1f}%**. To reach the next tier, focus on improving your {'attendance' if st.session_state.form_data['attendance'] < 90 else 'daily study hours'}.")
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "Study Planner":
    if st.session_state.results is None:
        st.warning("Please generate a plan first on the Home page.")
    else:
        st.title("📅 Personalized Study Schedule")
        timetable = generate_timetable(st.session_state.results['pred_score'])
        
        cols = st.columns(len(timetable))
        for i, (time, activity) in enumerate(timetable.items()):
            with cols[i]:
                st.markdown(f"""<div class="card" style='height: 100%; border-top: 5px solid #3b82f6;'>
                    <h4 style='color: #3b82f6;'>{time}</h4>
                    <p style='font-size: 1.1rem;'>{activity}</p>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🏅 Global Leaderboard (Top Peers)")
        if not df_full.empty:
            top_peers = df_full.sort_values(by='predicted_performance', ascending=False).head(5)
            st.dataframe(top_peers[['predicted_performance', 'study_hours', 'attendance']], use_container_width=True)

elif st.session_state.page == "Analytics":
    if st.session_state.results is None:
        st.warning("Please generate a plan first.")
    else:
        st.title("📈 Deep Learning Analytics")
        res = st.session_state.results
        
        if not df_full.empty:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                fig1 = px.scatter(df_full, x='study_hours', y='predicted_performance', 
                                 color='focus_level', size='attendance',
                                 title="Study Hours vs Performance",
                                 template="plotly_dark")
                fig1.add_trace(go.Scatter(x=[st.session_state.form_data['study_hours']], y=[res['pred_score']], 
                                         mode='markers', marker=dict(size=20, color='red', symbol='star'),
                                         name="YOU"))
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                importances = model.feature_importances_
                features = ['Study Hours', 'Sleep', 'Focus', 'Stress', 'Screen Time', 'Activity', 'Attendance', 'Prev Marks']
                fig2 = px.bar(x=features, y=importances, title="Impact Factors",
                             color=importances, color_continuous_scale='Blues',
                             template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Peer Group Benchmarking")
            fig3 = px.histogram(df_full, x='predicted_performance', color_discrete_sequence=['#3b82f6'],
                               title="Global Performance Distribution", template="plotly_dark")
            fig3.add_vline(x=res['pred_score'], line_width=3, line_dash="dash", line_color="red", annotation_text="YOU ARE HERE")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "AI Assistant":
    st.title("🤖 AI Study Companion")
    st.markdown("<p style='color: #94a3b8;'>Ask questions about your study plan, stress management, or productivity.</p>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I improve my focus?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            p = prompt.lower()
            if "timetable" in p or "schedule" in p:
                response = "I've crafted a personalized schedule for you in the 'Study Planner' tab. It balances hard subjects in the morning when your focus is highest!"
            elif "improve" in p or "score" in p:
                response = "To boost your predicted score, try increasing your study hours by 1 hour daily and practicing active recall."
            elif "stress" in p:
                response = "High stress levels detected! Try the 5-minute boxed breathing technique and ensure you get at least 7 hours of sleep."
            else:
                response = "That's an interesting point! Focusing on your weak areas identified in the Dashboard will yield the best results. Anything specific you'd like to dive into?"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif st.session_state.page == "Report":
    if st.session_state.results is None:
        st.warning("No data available to generate report. Please complete the assessment first.")
    else:
        st.title("📝 Performance Report")
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2991/2991108.png", width=150)
        st.header("Ready for Download")
        st.write("Your comprehensive AI-generated study report is ready. It includes your performance prediction, peer benchmarking, and personalized strategy.")
        
        try:
            res = st.session_state.results
            pdf_bytes = create_pdf(
                st.session_state.form_data['user_name'], 
                res['pred_score'], 
                res['category'], 
                res['recs'],
                st.session_state.form_data['user_id']
            )
            st.download_button(
                label="📥 Download Detailed PDF Report",
                data=pdf_bytes,
                file_name=f"Study_Planner_Report_{st.session_state.form_data['user_id']}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px; border-top: 1px solid rgba(255,255,255,0.05);'>
    Developed with ❤️ by Smart AI Team | Powered by Machine Learning & Streamlit
</div>
""", unsafe_allow_html=True)