import streamlit as st
import numpy as np
import pickle
import json
from PIL import Image
import google.generativeai as genai
from utils import preprocess_image, predict_caption, idx_to_word 

# ==========================================
# 1. PAGE SETUP & THEME
# ==========================================
st.set_page_config(page_title="AI Vision System | J.A.R.V.I.S", page_icon="👁️‍🗨️", layout="wide")

# CUSTOM CSS FOR CYBERPUNK / JARVIS HUD AESTHETIC
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

    /* Remove default Streamlit header and padding */
    header {visibility: hidden;}
    .main .block-container {padding-top: 1rem;}

    /* =========================================
       1. CYBER GRID BACKGROUND & FONTS
       ========================================= */
    .stApp {
        background-color: #030811;
        background-image: 
            linear-gradient(rgba(0, 243, 255, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.04) 1px, transparent 1px);
        background-size: 30px 30px;
    }

    /* Force all text to use tech fonts */
    .stMarkdown, p, label, .stSelectbox, .stRadio, .stCheckbox, li {
        color: #e0ffff !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 0.5px;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* =========================================
       2. HACKER/HUD CONTAINERS
       ========================================= */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(2, 10, 20, 0.85);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-top: 3px solid #00f3ff; /* Techy top border */
        border-radius: 4px;
        padding: 25px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.05), inset 0 0 20px rgba(0, 243, 255, 0.05);
        position: relative;
    }
    
    /* Corner tech accents */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown)::before {
        content: '';
        position: absolute;
        top: -1px; left: -1px;
        width: 15px; height: 15px;
        border-top: 2px solid #00f3ff;
        border-left: 2px solid #00f3ff;
    }

    /* =========================================
       3. GLOWING HERO TITLE (J.A.R.V.I.S Style)
       ========================================= */
    .hero-title {
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        text-align: center;
        margin-top: -30px;
        text-transform: uppercase;
        text-shadow: 0px 0px 5px rgba(0, 243, 255, 0.8), 0px 0px 15px rgba(0, 81, 255, 0.4);
        letter-spacing: 4px;
    }
    .hero-subtitle {
        text-align:center; 
        color:#00ffcc !important; 
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.2rem;
        margin-bottom: 30px;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* =========================================
       4. CYBER TERMINAL BUTTON
       ========================================= */
    .stButton>button {
        width: 100%;
        background-color: rgba(0, 243, 255, 0.05) !important;
        color: #00f3ff !important;
        border: 1px solid #00f3ff !important;
        border-radius: 0px !important; /* Sharp edges */
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.2), inset 0 0 10px rgba(0, 243, 255, 0.1);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00f3ff !important;
        color: #000000 !important;
        box-shadow: 0 0 20px #00f3ff, inset 0 0 20px #00f3ff;
        transform: scale(1.02);
    }
    
    /* Fixing the visibility of labels specifically */
    div[data-testid="stWidgetLabel"] p {
        color: #00ffcc !important; /* Brighter neon green/cyan */
        font-weight: 700 !important;
    }

    /* Inputs & Selectboxes styling */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(0, 20, 35, 0.8) !important;
        color: #00f3ff !important;
        border: 1px solid rgba(0, 243, 255, 0.4) !important;
        border-radius: 2px;
    }

    /* =========================================
       5. HUD KPI CARDS
       ========================================= */
    @keyframes hudScan {
        0% { border-left-color: #00f3ff; box-shadow: 0 0 10px rgba(0,243,255,0.1); }
        50% { border-left-color: #ff00ff; box-shadow: 0 0 20px rgba(0,243,255,0.4); }
        100% { border-left-color: #00f3ff; box-shadow: 0 0 10px rgba(0,243,255,0.1); }
    }

    .kpi-card {
        background: linear-gradient(90deg, rgba(0, 243, 255, 0.08) 0%, rgba(0, 0, 0, 0.4) 100%);
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-left: 4px solid #00f3ff;
        border-radius: 2px;
        padding: 20px;
        margin-bottom: 20px;
        position: relative;
        animation: hudScan 4s infinite alternate;
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        border: 1px solid #00f3ff;
        background: rgba(0, 243, 255, 0.1);
        transform: translateX(5px);
    }

    /* =========================================
       6. CYBER UPLOAD DROPZONE (FIXED)
       ========================================= */
    div[data-testid="stFileUploader"] > section {
        background-color: rgba(2, 10, 20, 0.8) !important;
        background-image: none !important; /* Forces the default purple to disappear */
        border: 1px dashed rgba(0, 243, 255, 0.5) !important;
        border-radius: 4px !important;
        padding: 20px !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    div[data-testid="stFileUploader"] > section:hover {
        border: 1px solid #00f3ff !important;
        background-color: rgba(0, 243, 255, 0.1) !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.4), inset 0 0 10px rgba(0, 243, 255, 0.2) !important;
    }

    div[data-testid="stFileUploader"] > section * {
        color: #00f3ff !important;
    }

    div[data-testid="stFileUploader"] button {
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: #00f3ff !important;
        border: 1px solid rgba(0, 243, 255, 0.5) !important;
        border-radius: 2px !important;
        text-transform: uppercase !important;
        font-family: 'Share Tech Mono', monospace !important;
        box-shadow: 0 0 5px rgba(0, 243, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background-color: #00f3ff !important;
        color: #000000 !important;
        box-shadow: 0 0 15px #00f3ff !important;
    }
       
    /* FLOATING CHATBOT CSS */
    .stChatFloating {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        width: 350px;
        background: rgba(2, 10, 20, 0.95);
        border: 1px solid #00f3ff;
        border-top: 4px solid #00f3ff;
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 0 20px rgba(0,243,255,0.2);
    }
    
    hr {
        border-top: 1px solid rgba(0, 243, 255, 0.3);
    }
    
    /* =========================================
       7. CYBER CHAT INPUT ZONE
       ========================================= */
    div[data-testid="stBottomBlockContainer"], 
    div[data-testid="stChatInput"] {
        background-color: #030811 !important; 
        background-image: none !important;
    }
    
    div[data-testid="stChatInput"] > div {
        background-color: rgba(2, 10, 20, 0.95) !important;
        border: 1px solid rgba(0, 243, 255, 0.4) !important;
        border-radius: 4px !important;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stChatInput"] > div:hover,
    div[data-testid="stChatInput"] > div:focus-within {
        border: 1px solid #00f3ff !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.4), inset 0 0 10px rgba(0, 243, 255, 0.1) !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #00f3ff !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    div[data-testid="stChatInput"] button {
        color: #00f3ff !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stChatInput"] button:hover {
        color: #ffffff !important;
        transform: scale(1.1);
        text-shadow: 0 0 10px #00f3ff !important;
    }

    </style>
    """, unsafe_allow_html=True)

# Clean, modern header
st.markdown('<h1 class="hero-title">SMART IMAGE DESCRIPTION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">[ Advanced Multimodal Analysis HUD ]</p>', unsafe_allow_html=True)

# ==========================================
# 2. API & MODEL SETUP
# ==========================================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

# INITIALIZE CHATBOT STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 3. IMAGE UPLOAD, CAMERA & COMPARISON
# ==========================================
col_a, col_b = st.columns(2)

image = None 

with col_a:
    st.markdown('<h3 style="color:#00f3ff; text-transform:uppercase;">[ 📸 Main Input ]</h3>', unsafe_allow_html=True)
    input_mode = st.radio("Select Data Source:", ["Upload File", "System Camera"], horizontal=True)
    
    if input_mode == "Upload File":
        uploaded_file = st.file_uploader("Initialize Uplink (Image)", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
    else:
        camera_file = st.camera_input("Activate Camera Sensor")
        if camera_file:
            image = Image.open(camera_file)
            st.download_button(
                label="[📥] Extract Capture Data",
                data=camera_file,
                file_name="sensor_capture.png",
                mime="image/png"
            )

    if image:
        st.image(image, caption="Current Visual Data", use_container_width=True)

with col_b:
    st.markdown('<h3 style="color:#00f3ff; text-transform:uppercase;">[ 🔄 Comparison Mode ]</h3>', unsafe_allow_html=True)
    enable_comparison = st.checkbox("Initialize Dual-Image Comparison")
    uploaded_file_2 = None
    if enable_comparison:
        uploaded_file_2 = st.file_uploader("Upload Secondary Reference", type=['jpg', 'jpeg', 'png'])
        if uploaded_file_2:
            image_2 = Image.open(uploaded_file_2)
            st.image(image_2, caption="Reference Data", use_container_width=True)

# ==========================================
# 4. CONFIGURATION PANEL
# ==========================================
st.divider()
st.markdown('<h3 style="color:#00f3ff; text-transform:uppercase;">[ ⚙️ Analysis Parameters ]</h3>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    language = st.selectbox("1. Output Language", ["English", "Bengali", "Hindi"])
    style = st.selectbox("2. Caption Style", ["Basic", "Professional", "Marketing", "Social media style"])

with c2:
    story_mode = st.selectbox("4. Story Mode", ["Off", "Short Story", "Medium Story", "Long Story"])
    add_hashtags = st.checkbox("5. Hashtags")
    detect_mood = st.checkbox("6. Emotion-Detection")

with c3:
    detect_quality = st.checkbox("9. Image Quality Resulation")
    crop_suggest = st.checkbox("10. Social Media Crop Suggestion")

user_qa = st.text_input("8. Q&A: Ask a Question about your ", placeholder="e.g., Target identified?")

# ==========================================
# 5. EXECUTION ENGINE
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button(">> RUN FULL AI ANALYSIS <<"):
    if image is None: 
        st.warning("ERR: Visual data not found. Please upload image.")
    else:
        with st.spinner("Compiling visual data... running neural networks..."):
            try:
                img_input = [image.convert('RGB')]
                if enable_comparison and uploaded_file_2:
                    img_input.append(image_2.convert('RGB'))

                prompt = f"""
                Act as an expert vision AI. Provide a detailed report in {language}. 
                You MUST return the output STRICTLY as a valid JSON object. Do not use markdown blocks like ```json.
                CRITICAL RULE: All values MUST be simple text strings. Do NOT create nested JSON objects or dictionaries inside the values. Avoid using unescaped double quotes inside the string values.
                ONLY include keys if the user requested them. If not requested, leave the value empty "".
                
                Format:
                {{
                    "caption": "Generate a {style} caption.",
                    "mood": "{ 'Analyze and describe the emotion/mood of the scene.' if detect_mood else '' }",
                    "story": "{ f'Write a {story_mode} based on visual elements.' if story_mode != 'Off' else '' }",
                    "quality": "{ 'Assess image quality.' if detect_quality else '' }",
                    "social": "{ 'Provide #hashtags and crop sizes as a single plain text paragraph. DO NOT use nested json.' if (add_hashtags or crop_suggest) else '' }",
                    "qa": "{ f'Answer: {user_qa}' if user_qa else '' }",
                    "comparison": "{ 'Compare the images.' if enable_comparison else '' }"
                }}
                """

                response = gemini_model.generate_content([prompt] + img_input)
                
                try:
                    clean_text = response.text.replace('```json', '').replace('```', '').strip()
                    result_data = json.loads(clean_text)
                    
                    st.markdown("---")
                    st.markdown('<h3 style="color:#00f3ff; text-align:center; text-transform:uppercase; text-shadow: 0 0 10px #00f3ff;">>> ANALYSIS HUD ACTIVE <<</h3>', unsafe_allow_html=True)
                    
                    def kpi_card(title, icon, content):
                        st.markdown(f"""
                        <div class="kpi-card">
                            <h4 style="color: #00f3ff; margin-top: 0; font-family: 'Orbitron', sans-serif; font-size: 1.1rem; border-bottom: 1px solid rgba(0,243,255,0.3); padding-bottom: 8px;">
                                <span style="font-size: 1.2rem; margin-right: 8px;">{icon}</span> {title}
                            </h4>
                            <p style="color: #e0ffff; font-size: 1rem; line-height: 1.5; margin-bottom: 0;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    r_col1, r_col2, r_col3 = st.columns(3)

                    with r_col1:
                        if result_data.get("caption"): kpi_card(f"Log: Caption ({style})", "📝", result_data["caption"])
                        if result_data.get("social"): kpi_card("Log: Social-Media Kit", "🏷️", result_data["social"])
                        
                    with r_col2:
                        if result_data.get("mood"): kpi_card("Log: Mood-Detection", "🎭", result_data["mood"])
                        if result_data.get("story"): kpi_card("Log: Visual-Story", "📖", result_data["story"])
                        if result_data.get("comparison"): kpi_card("Log: Image-Comparison", "🔄", result_data["comparison"])
                        
                    with r_col3:
                        if result_data.get("quality"): kpi_card("Log: Diagnostic-Quality Check", "🔍", result_data["quality"])
                        if result_data.get("qa"): kpi_card("Log: Q&A Answer", "🤖", result_data["qa"])

                except json.JSONDecodeError:
                    st.warning("ERR: Display module failure. Accessing raw AI core data:")
                    st.write(response.text)

            except Exception as e:
                st.error(f"SYSTEM FAILURE: {e}")

# ==========================================
# 6. DYNAMIC CHATBOT 
# ==========================================
st.divider()
st.markdown('<h3 style="color:#00f3ff; text-transform:uppercase;">[ 🤖 AI Chat-Link ]</h3>', unsafe_allow_html=True)

chat_display_area = st.container()

with chat_display_area:
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if chat_user_input := st.chat_input("Enter command or query..."):
    st.session_state.chat_history.append({"role": "user", "content": chat_user_input})
    
    with chat_display_area:
        with st.chat_message("user"):
            st.markdown(chat_user_input)

        with st.chat_message("assistant"):
            try:
                chat_response = gemini_model.generate_content(chat_user_input)
                st.markdown(chat_response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": chat_response.text})
            except Exception as e:
                st.error(f"AI Chat-Link Error: {e}")
