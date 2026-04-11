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

# ------------------ CUSTOM CSS (Premium UI) ------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #ec4899;
        --background: #0f172a;
        --surface: rgba(255, 255, 255, 0.05);
        --text: #f8fafc;
        --accent: #10b981;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        color: var(--text);
    }

    /* Glassmorphism Card */
    .glass-card {
        background: var(--surface);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 32px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        transition: transform 0.3s ease, border 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 30px;
    }

    .metric-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white !important;
        border: none !important;
        padding: 16px 32px !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.4) !important;
    }

    /* Horizontal Nav */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        padding: 20px 0;
        margin-bottom: 40px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 100px;
    }

    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        color: #94a3b8;
        padding: 0 24px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
if 'page' not in st.session_state:
    st.session_state.page = "Input"

if 'results' not in st.session_state:
    st.session_state.results = None

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'name': "Guest Student",
        'id': "ST-001",
        'study_hours': 5.0,
        'sleep_hours': 7.0,
        'focus_level': 7,
        'stress_level': 4,
        'screen_time': 3.0,
        'activity': 1,
        'attendance': 85,
        'prev_marks': 75
    }

# ------------------ DATA & MODEL LOADING ------------------
@st.cache_resource
def load_all_models():
    """Load ML models with robust error handling"""
    model_files = {
        "model": "student_performance_model.pkl",
        "scaler": "scaler.pkl",
        "clusters": "student_clusters.pkl",
        "accuracy": "model_accuracy.pkl"
    }
    
    loaded_objects = {}
    missing_files = []
    
    for key, filename in model_files.items():
        if os.path.exists(filename):
            try:
                loaded_objects[key] = joblib.load(filename)
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {e}")
                return None
        else:
            missing_files.append(filename)
            
    if missing_files:
        st.warning(f"⚠️ Missing model files: {', '.join(missing_files)}. Please run 'train_model.py' first.")
        return None
        
    return loaded_objects

@st.cache_data
def get_dataset():
    """Load historical data for benchmarking"""
    if os.path.exists("data.csv"):
        try:
            return pd.read_csv("data.csv")
        except:
            return pd.DataFrame()
    return pd.DataFrame()

# Initialize models and data
models = load_all_models()
df_historical = get_dataset()

# ------------------ UTILS ------------------
def classify_rank(score):
    if score >= 85: return "Diamond", "💎", "#10b981"
    if score >= 70: return "Gold", "🥇", "#f59e0b"
    if score >= 50: return "Silver", "🥈", "#6366f1"
    return "Bronze", "🥉", "#ef4444"

def generate_recommendations(data, score):
    recs = []
    if score < 60:
        recs.append("🔴 Focus on core fundamentals through active recall.")
    if data['sleep_hours'] < 7:
        recs.append("😴 Increase sleep to 7.5+ hours for memory consolidation.")
    if data['focus_level'] < 6:
        recs.append("🧠 Use Pomodoro technique (25/5 breaks) to boost focus.")
    if data['stress_level'] > 6:
        recs.append("🧘 Include 15 mins of mindfulness or physical activity daily.")
    if data['study_hours'] < 4:
        recs.append("📖 Gradually increase study blocks by 30 mins each week.")
    
    if not recs:
        recs.append("✅ You are on a great path! Keep the consistency.")
    return recs

# ------------------ UI NAVIGATION ------------------
# We use a simple horizontal menu simulation
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🧠 SMART AI STUDY PLANNER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 30px;'>Advanced Student Performance Prediction & Learning Optimization</p>", unsafe_allow_html=True)

# Select Page
tabs = st.tabs(["📋 Assessment", "📊 Insight Dashboard", "📈 Analytics & Growth"])

# ------------------ PAGE 1: INPUT ------------------
with tabs[0]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📝 Personalized Assessment")
    st.write("Complete your daily metrics to generate your AI-powered study roadmap.")
    
    with st.form("assessment_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 👤 Profile")
            u_name = st.text_input("Full Name", value=st.session_state.form_data['name'])
            u_id = st.text_input("Student ID", value=st.session_state.form_data['id'])
            
            st.markdown("#### 📖 Academic Habit")
            u_study = st.slider("Daily Study Hours", 1.0, 12.0, float(st.session_state.form_data['study_hours']))
            u_sleep = st.slider("Daily Sleep Hours", 4.0, 10.0, float(st.session_state.form_data['sleep_hours']))
            u_focus = st.select_slider("Focus Level", options=list(range(1, 11)), value=int(st.session_state.form_data['focus_level']))
            
        with c2:
            st.markdown("#### ⚖️ Lifestyle Metrics")
            u_stress = st.select_slider("Stress Level", options=list(range(1, 11)), value=int(st.session_state.form_data['stress_level']))
            u_screen = st.slider("Non-Study Screen Time (Hours)", 0.0, 10.0, float(st.session_state.form_data['screen_time']))
            u_activity = st.selectbox("Regular Physical Activity?", options=[0, 1], index=st.session_state.form_data['activity'], format_func=lambda x: "Yes" if x==1 else "No")
            
            st.markdown("#### 🎓 Current Standing")
            u_attendance = st.number_input("Attendance %", 0, 100, int(st.session_state.form_data['attendance']))
            u_prev = st.number_input("Previous Marks %", 0, 100, int(st.session_state.form_data['prev_marks']))

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("✨ GENERATE AI PLANNER")
        
        if submit_btn:
            if models is None:
                st.error("❌ Models are not loaded. Please contact administrator or run training script.")
            else:
                # Update Session State
                st.session_state.form_data = {
                    'name': u_name, 'id': u_id, 'study_hours': u_study,
                    'sleep_hours': u_sleep, 'focus_level': u_focus,
                    'stress_level': u_stress, 'screen_time': u_screen,
                    'activity': u_activity, 'attendance': u_attendance,
                    'prev_marks': u_prev
                }
                
                # Prepare Input for Prediction
                input_data = pd.DataFrame([{
                    'study_hours': u_study,
                    'sleep_hours': u_sleep,
                    'focus_level': u_focus,
                    'stress_level': u_stress,
                    'screen_time': u_screen,
                    'physical_activity': u_activity,
                    'attendance': u_attendance,
                    'previous_marks': u_prev
                }])
                
                # Run Inference
                try:
                    pred_score = models['model'].predict(input_data)[0]
                    scaled_data = models['scaler'].transform(input_data)
                    cluster = models['clusters'].predict(scaled_data)[0]
                    
                    st.session_state.results = {
                        'score': pred_score,
                        'cluster': cluster,
                        'input_df': input_data,
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    st.success("✅ Prediction successful! Switch to the 'Dashboard' tab.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Inference Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ PAGE 2: DASHBOARD ------------------
with tabs[1]:
    if st.session_state.results is None:
        st.info("👋 Please complete the 'Assessment' first to see your results.")
    else:
        res = st.session_state.results
        data = st.session_state.form_data
        rank, icon, r_color = classify_rank(res['score'])
        
        # Header Stats
        st.markdown(f"### 📊 Analysis for {data['name']} (ID: {data['id']})")
        
        # Card Layout for Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-label'>Predicted Score</div>
                <div class='metric-value' style='background: linear-gradient(135deg, {r_color}, #fff); -webkit-background-clip: text;'>{res['score']:.1f}%</div>
                <div style='color: {r_color}; font-weight: 600;'>{icon} {rank} Tier</div>
            </div>""", unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-label'>Efficiency Index</div>
                <div class='metric-value'>{data['focus_level']*10}%</div>
                <div style='color: #94a3b8;'>Based on Focus Level</div>
            </div>""", unsafe_allow_html=True)
            
        with col_m3:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-label'>Peer Cluster</div>
                <div class='metric-value'>#{res['cluster']}</div>
                <div style='color: #a855f7;'>K-Means Group</div>
            </div>""", unsafe_allow_html=True)
            
        with col_m4:
            acc = models['accuracy'] * 100 if models else 0
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-label'>Model Reliability</div>
                <div class='metric-value'>{acc:.1f}%</div>
                <div style='color: #10b981;'>R² Confidence</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        cc1, cc2 = st.columns([1, 1])
        
        with cc1:
            st.markdown("#### 🎯 Performance Radar")
            # Create a simple gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted Final Score", 'font': {'size': 20, 'color': '#fff'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#fff"},
                    'bar': {'color': r_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [50, 85], 'color': 'rgba(99, 102, 241, 0.1)'},
                        {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Outfit"}, height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with cc2:
            st.markdown("#### 💡 AI Recommendations")
            recs = generate_recommendations(data, res['score'])
            for r in recs:
                st.markdown(f"<div style='background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid {r_color};'>{r}</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ PAGE 3: ANALYTICS ------------------
with tabs[2]:
    if st.session_state.results is None:
        st.info("👋 Analytics will appear once an Assessment is completed.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🚀 Strategic Growth Insights")
        
        if not df_historical.empty:
            c11, c22 = st.columns(2)
            
            with c11:
                st.markdown("##### 📍 Your Position in Peer Network")
                fig_scatter = px.scatter(
                    df_historical, 
                    x='study_hours', 
                    y='predicted_performance',
                    color='focus_level',
                    size='attendance',
                    template='plotly_dark',
                    color_continuous_scale='Magma'
                )
                fig_scatter.add_trace(go.Scatter(
                    x=[st.session_state.form_data['study_hours']], 
                    y=[st.session_state.results['score']],
                    mode='markers',
                    marker=dict(size=25, color='cyan', symbol='star', line=dict(width=2, color='white')),
                    name='YOU'
                ))
                fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with c22:
                st.markdown("##### 📊 Key Success Factors")
                if hasattr(models['model'], 'feature_importances_'):
                    feat_imp = models['model'].feature_importances_
                    features = ['Study', 'Sleep', 'Focus', 'Stress', 'Screen', 'Activity', 'Attendance', 'Record']
                    fig_imp = px.bar(
                        x=features, y=feat_imp, 
                        labels={'x': 'Feature', 'y': 'Importance'},
                        color=feat_imp,
                        color_continuous_scale='Viridis',
                        template='plotly_dark'
                    )
                    fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("No historical data found for benchmarking. Displaying simulated impact analysis.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Built by Smart AI Team | Premium AI Student Planner Dashboard</p>", unsafe_allow_html=True)