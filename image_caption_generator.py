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
st.set_page_config(page_title="Multi-Feature AI Vision", page_icon="📸", layout="wide")

# CUSTOM CSS FOR VISIBILITY, REMOVING PADDING & FLOATING ANIMATIONS
st.markdown("""
    <style>
    /* Remove default Streamlit header and padding */
    header {visibility: hidden;}
    .main .block-container {padding-top: 2rem;}

    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Force all text to be visible */
    .stMarkdown, p, label, .stSelectbox, .stRadio, .stCheckbox {
        color: #ffffff !important;
    }

    /* Glassmorphism Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Title Styling */
    .hero-title {
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-top: -50px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white !important;
        border: none;
        font-weight: bold;
        padding: 10px;
    }
    
    /* Fixing the visibility of labels specifically */
    div[data-testid="stWidgetLabel"] p {
        color: #00d2ff !important;
        font-weight: 600 !important;
    }

    /* =========================================
       FLOATING MOTION CSS FOR KPI CARDS (হাওয়াতে ভাসবে)
       ========================================= */
    @keyframes floatingEffect {
        0% {
            transform: translateY(0px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        50% {
            transform: translateY(-12px); /* হাল্কা উপরে উঠবে */
            box-shadow: 0 15px 20px rgba(0, 210, 255, 0.2); /* শ্যাডো বাড়বে */
        }
        100% {
            transform: translateY(0px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    }

    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0, 210, 255, 0.3);
        margin-bottom: 20px;
        /* ৩ সেকেন্ড ধরে ইনফিনিট লুপে ভাসবে */
        animation: floatingEffect 3s ease-in-out infinite;
        transition: border 0.3s ease, background 0.3s ease;
    }

    /* Hover Effect - হোভার করলে বর্ডার আর ব্যাকগ্রাউন্ড একটু উজ্জ্বল হবে */
    .kpi-card:hover {
        border: 1px solid rgba(0, 210, 255, 0.8);
        background: rgba(255, 255, 255, 0.1);
    }
            
    /* FLOATING CHATBOT CSS */
    .stChatFloating {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        width: 350px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(0, 210, 255, 0.3);
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Clean, modern header without the extra boxes
st.markdown('<h1 class="hero-title">Smart Image Description System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#a1a1a1;">Advanced Multimodal Image Analysis</p>', unsafe_allow_html=True)

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
    st.subheader("📸 Main Input")
    input_mode = st.radio("Choose Input Method:", ["Upload File", "Use Camera"], horizontal=True)
    
    if input_mode == "Upload File":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file:
            image = Image.open(camera_file)
            st.download_button(
                label="📥 Download Capture",
                data=camera_file,
                file_name="camera_capture.png",
                mime="image/png"
            )

    if image:
        st.image(image, caption="Current Selection", use_container_width=True)

with col_b:
    st.subheader("🔄 Comparison Mode")
    enable_comparison = st.checkbox("Enable Image Comparison Mode")
    uploaded_file_2 = None
    if enable_comparison:
        uploaded_file_2 = st.file_uploader("Upload Second Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file_2:
            image_2 = Image.open(uploaded_file_2)
            st.image(image_2, caption="Comparison Image", use_container_width=True)

# ==========================================
# 4. CONFIGURATION PANEL
# ==========================================
st.divider()
st.subheader("⚙️ Analysis Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    language = st.selectbox("1. Language", ["English", "Bengali", "Hindi"])
    style = st.selectbox("2. Caption Style", ["Basic", "Professional", "Marketing", "Social media style"])

with c2:
    story_mode = st.selectbox("4. Story Mode", ["Off", "Short Story", "Medium Story", "Long Story"])
    add_hashtags = st.checkbox("5. Hashtag Generator")
    detect_mood = st.checkbox("6. Mood Detection")

with c3:
    detect_quality = st.checkbox("9. Image Quality Detection")
    crop_suggest = st.checkbox("10. Social Media Crop Suggestions")

user_qa = st.text_input("8. Q&A: Ask a question about the image", placeholder="e.g., What is written on the sign?")

# ==========================================
# 5. EXECUTION ENGINE
# ==========================================
if st.button("🚀 Run Full AI Analysis"):
    if image is None: 
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Analyzing parameters and generating report..."):
            try:
                img_input = [image.convert('RGB')]
                if enable_comparison and uploaded_file_2:
                    img_input.append(image_2.convert('RGB'))

                # Dynamic Prompt based on UI Parameters with STRICT JSON RULES
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
                    # Clean the JSON response just in case
                    clean_text = response.text.replace('```json', '').replace('```', '').strip()
                    result_data = json.loads(clean_text)
                    
                    st.markdown("---")
                    st.subheader("📊 Dynamic Analysis Dashboard")
                    
                    # Animated Floating KPI Card Function
                    def kpi_card(title, icon, content, delay=0):
                        # `animation-delay` ensures the cards don't float up and down at the exact same time
                        st.markdown(f"""
                        <div class="kpi-card" style="animation-delay: {delay}s;">
                            <h4 style="color: #00d2ff; margin-top: 0; font-size: 1.1rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                                {icon} {title}
                            </h4>
                            <p style="color: #e0e0e0; font-size: 1rem; line-height: 1.5; margin-bottom: 0;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Dynamic Layout - Only render cards if data exists
                    r_col1, r_col2, r_col3 = st.columns(3)
                    
                    # Each card gets a slightly different delay so they float organically
                    anim_delay = 0.0 

                    with r_col1:
                        if result_data.get("caption"): 
                            kpi_card(f"Caption ({style})", "📝", result_data["caption"], anim_delay)
                            anim_delay += 0.5
                        if result_data.get("social"): 
                            kpi_card("Social Media Kit", "🏷️", result_data["social"], anim_delay)
                            anim_delay += 0.5
                        
                    with r_col2:
                        if result_data.get("mood"): 
                            kpi_card("Mood Detection", "🎭", result_data["mood"], anim_delay)
                            anim_delay += 0.5
                        if result_data.get("story"): 
                            kpi_card(f"Visual Story", "📖", result_data["story"], anim_delay)
                            anim_delay += 0.5
                        if result_data.get("comparison"): 
                            kpi_card("Image Comparison", "🔄", result_data["comparison"], anim_delay)
                            anim_delay += 0.5
                        
                    with r_col3:
                        if result_data.get("quality"): 
                            kpi_card("Quality Check", "🔍", result_data["quality"], anim_delay)
                            anim_delay += 0.5
                        if result_data.get("qa"): 
                            kpi_card("Q&A Answer", "🤖", result_data["qa"], anim_delay)
                            anim_delay += 0.5

                except json.JSONDecodeError:
                    st.warning("⚠️ Data formatting issue. Showing raw AI output instead:")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# ==========================================
# 6. DYNAMIC CHATBOT 
# ==========================================
st.divider()
st.markdown("### 🤖 AI Chat Assistant")

chat_display_area = st.container()

with chat_display_area:
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if chat_user_input := st.chat_input("Ask me anything about the images..."):
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
                st.error(f"Chat Error: {e}")
