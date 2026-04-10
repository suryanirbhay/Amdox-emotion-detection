import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from emotion_model import predict_emotion
from recommendation import recommend_task
from analytics import calculate_stress_score, employee_risk_level
from facial_emotion_detection import detect_face_emotion, draw_faces
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Amdox - Emotion Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS for industry-level styling
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main {
        background-color: #ffffff;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    .header-container h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        font-size: 1.1em;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    /* Section Headers */
    h2 {
        color: #1a202c;
        font-size: 1.8em;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    h3 {
        color: #2d3748;
        font-size: 1.3em;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 10px 15px;
        font-size: 1em;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
    }
    
    /* Alert Styling */
    .stSuccess {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        border-radius: 8px;
    }
    
    .stWarning {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: #dbeafe;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Setup database
@st.cache_resource
def get_connection():
    conn = sqlite3.connect("mood_data.db", check_same_thread=False)
    return conn

@st.cache_resource
def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Drop existing table if it has wrong schema
    try:
        cursor.execute("PRAGMA table_info(mood_tracker)")
        columns = cursor.fetchall()
        if len(columns) != 6:
            cursor.execute("DROP TABLE IF EXISTS mood_tracker")
    except:
        pass
    
    # Create table with correct schema
    conn.execute('''
        CREATE TABLE IF NOT EXISTS mood_tracker (
            employee_id TEXT,
            text TEXT,
            emotion TEXT,
            confidence REAL,
            stress_score INTEGER,
            date TEXT
        )
    ''')
    conn.commit()
    return conn

conn = init_db()

# Header Section
st.markdown("""
<div class="header-container">
    <h1>🧠 Amdox AI-Powered Emotion Intelligence System</h1>
    <p class="header-subtitle">Real-time emotional analysis and employee wellness management</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for input
st.sidebar.markdown("### 📝 Employee Feedback")
st.sidebar.markdown("---")

employee_id = st.sidebar.text_input("🆔 Employee ID", placeholder="e.g., EMP001")
user_text = st.sidebar.text_area("💬 Your Message", placeholder="Share how you're feeling...", height=120)

col1, col2 = st.sidebar.columns([2, 1])
with col1:
    analyze_btn = st.sidebar.button("🔍 Analyze Emotion", use_container_width=True)
with col2:
    st.sidebar.write("")

if analyze_btn:
    if employee_id and user_text:
        try:
            with st.spinner("🔄 Analyzing emotion..."):
                emotion, confidence = predict_emotion(user_text)
                stress_score = calculate_stress_score(emotion)

                conn.execute("INSERT INTO mood_tracker VALUES (?, ?, ?, ?, ?, ?)",
                             (employee_id, user_text, emotion, float(confidence),
                              stress_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()

            st.sidebar.success("✅ Analysis saved successfully!")
            
            # Results Section
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            emotion_colors = {
                "happy": "#10b981",
                "sad": "#3b82f6",
                "neutral": "#6b7280",
                "stressed": "#ef4444",
                "angry": "#dc2626"
            }
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {emotion_colors.get(emotion, '#667eea')}">
                    <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Detected Emotion</h4>
                    <p style="color: {emotion_colors.get(emotion, '#667eea')}; font-size: 1.8em; font-weight: 700;">{emotion.upper()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #3b82f6;">
                    <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Confidence</h4>
                    <p style="color: #3b82f6; font-size: 1.8em; font-weight: 700;">{confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                stress_color = "#ef4444" if stress_score >= 2 else "#f59e0b" if stress_score == 1 else "#10b981"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {stress_color};">
                    <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Stress Level</h4>
                    <p style="color: {stress_color}; font-size: 1.8em; font-weight: 700;">{stress_score}/3</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                task = recommend_task(emotion)
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #667eea;">
                    <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Recommendation</h4>
                    <p style="color: #667eea; font-size: 0.95em; font-weight: 600;">{task}</p>
                </div>
                """, unsafe_allow_html=True)

            if emotion == "stressed":
                st.error("⚠️ Alert: Employee is showing stress signs. Consider follow-up support.")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
    else:
        st.warning("⚠️ Please enter both Employee ID and Message")

# FACIAL EMOTION DETECTION
st.sidebar.markdown("---")
st.sidebar.markdown("### 😊 Facial Emotion Detection")

facial_option = st.sidebar.radio("Choose input method:", ["Upload Image", "Take Photo"], key="facial_option")

facial_emotion = None
facial_confidence = None
face_detected = False

if facial_option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.sidebar.button("🔍 Analyze Face", use_container_width=True):
            with st.spinner("🔄 Analyzing facial expression..."):
                facial_emotion, facial_confidence, faces_count = detect_face_emotion(image)
                if facial_emotion:
                    face_detected = True
                else:
                    st.sidebar.error("❌ No face detected in the image. Please try another image.")
else:
    # Photo capture option
    st.sidebar.write("📸 Take a photo with your camera")
    picture = st.camera_input("Capture your face", key="camera_input")
    
    if picture is not None:
        if st.sidebar.button("🔍 Analyze Face", use_container_width=True):
            with st.spinner("🔄 Analyzing facial expression..."):
                image = Image.open(picture)
                facial_emotion, facial_confidence, faces_count = detect_face_emotion(image)
                if facial_emotion:
                    face_detected = True
                else:
                    st.sidebar.error("❌ No face detected. Please try again.")

# Display facial emotion results
if face_detected and facial_emotion:
    st.markdown("---")
    st.markdown("### 😊 Facial Expression Analysis")
    
    emotion_colors = {
        "happy": "#10b981",
        "sad": "#3b82f6",
        "neutral": "#6b7280",
        "stressed": "#ef4444",
        "angry": "#dc2626"
    }
    
    fcol1, fcol2, fcol3 = st.columns(3)
    
    with fcol1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {emotion_colors.get(facial_emotion, '#667eea')}">
            <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Facial Emotion</h4>
            <p style="color: {emotion_colors.get(facial_emotion, '#667eea')}; font-size: 1.8em; font-weight: 700;">{facial_emotion.upper()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with fcol2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3b82f6;">
            <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Confidence</h4>
            <p style="color: #3b82f6; font-size: 1.8em; font-weight: 700;">{facial_confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with fcol3:
        # Get recommendation based on facial emotion
        facial_task = recommend_task(facial_emotion)
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #667eea;">
            <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Recommendation</h4>
            <p style="color: #667eea; font-size: 0.95em; font-weight: 600;">{facial_task}</p>
        </div>
        """, unsafe_allow_html=True)


# HR DASHBOARD
st.markdown("---")
st.markdown("## 📈 HR Analytics Dashboard")
st.markdown("")

try:
    data = pd.read_sql_query("SELECT * FROM mood_tracker", conn)

    if not data.empty:
        # Key Metrics Section
        st.markdown("### Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #667eea;">
                <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Total Records</h4>
                <p style="color: #667eea; font-size: 2em; font-weight: 700;">{len(data)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            stressed_count = len(data[data['emotion'] == 'stressed'])
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #ef4444;">
                <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Stressed Entries</h4>
                <p style="color: #ef4444; font-size: 2em; font-weight: 700;">{stressed_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            happy_count = len(data[data['emotion'] == 'happy'])
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #10b981;">
                <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Happy Entries</h4>
                <p style="color: #10b981; font-size: 2em; font-weight: 700;">{happy_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_stress = data['stress_score'].mean()
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Avg Stress Score</h4>
                <p style="color: #f59e0b; font-size: 2em; font-weight: 700;">{avg_stress:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Visualizations Section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎭 Emotion Distribution")
            emotion_counts = data['emotion'].value_counts()
            
            colors_map = {
                "happy": "#10b981",
                "sad": "#3b82f6",
                "neutral": "#6b7280",
                "stressed": "#ef4444",
                "angry": "#dc2626"
            }
            
            fig_emotion = go.Figure(data=[
                go.Bar(
                    x=emotion_counts.index,
                    y=emotion_counts.values,
                    marker=dict(
                        color=[colors_map.get(e, '#667eea') for e in emotion_counts.index],
                        line=dict(color='white', width=2)
                    ),
                    text=emotion_counts.values,
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )
            ])
            fig_emotion.update_layout(
                title="",
                xaxis_title="Emotion Type",
                yaxis_title="Count",
                template='plotly_white',
                hovermode='x unified',
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_emotion, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Stress Trend Over Time")
            stress_trend = data.groupby("date")["stress_score"].mean().reset_index()
            
            fig_stress = go.Figure(data=[
                go.Scatter(
                    x=stress_trend['date'],
                    y=stress_trend['stress_score'],
                    mode='lines+markers',
                    name='Avg Stress Score',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8, symbol='circle'),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    hovertemplate='<b>%{x}</b><br>Stress: %{y:.2f}<extra></extra>'
                )
            ])
            fig_stress.update_layout(
                title="",
                xaxis_title="Date",
                yaxis_title="Stress Score",
                template='plotly_white',
                hovermode='x unified',
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_stress, use_container_width=True)

        st.markdown("---")

        if len(data['employee_id'].unique()) > 0:
            st.markdown("### 👥 Employee Risk Assessment")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_emp = st.selectbox("📍 Select Employee to Review", sorted(data['employee_id'].unique()))
            
            emp_data = data[data['employee_id'] == selected_emp]
            if not emp_data.empty:
                risk = employee_risk_level(emp_data, selected_emp)
                avg_emp_stress = emp_data['stress_score'].mean()
                
                risk_color_map = {
                    "HIGH RISK": "#ef4444",
                    "MODERATE RISK": "#f59e0b",
                    "LOW RISK": "#10b981"
                }
                
                col_risk1, col_risk2, col_risk3 = st.columns(3)
                
                with col_risk1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {risk_color_map.get(risk, '#667eea')};">
                        <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Risk Level</h4>
                        <p style="color: {risk_color_map.get(risk, '#667eea')}; font-size: 1.5em; font-weight: 700;">{risk}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_risk2:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: #f59e0b;">
                        <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Avg Stress</h4>
                        <p style="color: #f59e0b; font-size: 1.5em; font-weight: 700;">{avg_emp_stress:.2f}/3</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_risk3:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: #3b82f6;">
                        <h4 style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Entries</h4>
                        <p style="color: #3b82f6; font-size: 1.5em; font-weight: 700;">{len(emp_data)}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 Recent Records")
        
        display_data = data.tail(10).copy()
        display_data = display_data[['employee_id', 'emotion', 'confidence', 'stress_score', 'date']]
        display_data.columns = ['Employee ID', 'Emotion', 'Confidence (%)', 'Stress Score', 'Date']
        
        st.dataframe(display_data, use_container_width=True, hide_index=True)

    else:
        st.info("📭 No data available yet. Start by entering employee feedback in the sidebar!")

except Exception as e:
    st.error(f"❌ Error loading dashboard: {str(e)}")
