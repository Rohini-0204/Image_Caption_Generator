import streamlit as st
import numpy as np
import pickle
from PIL import Image
import google.generativeai as genai
import tensorflow as tf
from huggingface_hub import hf_hub_download
from utils import preprocess_image, predict_caption, idx_to_word 

# ==========================================
# 0. HUGGING FACE MODEL LOADING (NEW)
# ==========================================
@st.cache_resource
def load_hf_model():
    try:
        # Hugging Face theke model weights download kora
        model_path = hf_hub_download(repo_id="rohini0204/Image_Cap_Genrator", filename="best_model.h5")
        # Hugging Face theke tokenizer/features pkl download kora
        tokenizer_path = hf_hub_download(repo_id="rohini0204/Image_Cap_Genrator", filename="features.pkl")
        
        # Model loading
        model = tf.keras.models.load_model(model_path)
        
        # Tokenizer loading
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None, None

# Model initialize kora
model, tokenizer = load_hf_model()

# ==========================================
# 1. PAGE SETUP & THEME
# ==========================================
st.set_page_config(page_title="Multi-Feature AI Vision", page_icon="📸", layout="wide")

st.markdown("""
    <style>
    header {visibility: hidden;}
    .main .block-container {padding-top: 2rem;}
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .stMarkdown, p, label, .stSelectbox, .stRadio, .stCheckbox {
        color: #ffffff !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-top: -50px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white !important;
        border: none;
        font-weight: bold;
        padding: 10px;
    }
    div[data-testid="stWidgetLabel"] p {
        color: #00d2ff !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">Smart Image Description System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#a1a1a1;">Advanced Multimodal Image Analysis</p>', unsafe_allow_html=True)

# ==========================================
# 2. API & MODEL SETUP (GEMINI)
# ==========================================
# Make sure GEMINI_API_KEY is set in your Streamlit secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
else:
    st.error("Gemini API Key missing in Secrets!")

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

    if image:
        st.image(image, caption="Current Selection", use_container_width=True)

with col_b:
    st.subheader("🔄 Comparison Mode")
    enable_comparison = st.checkbox("Enable Image Comparison Mode")
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
        with st.spinner("Processing..."):
            try:
                img_input = [image.convert('RGB')]
                if enable_comparison and 'image_2' in locals():
                    img_input.append(image_2.convert('RGB'))

                prompt = f"""
                Act as an expert vision AI. Provide a detailed report in {language}. 
                Format the output strictly with the following bold headings:

                **CAPTION:**
                Generate a {style} caption for the primary image.

                **MOOD DETECTION:**
                { "Analyze and describe the emotion/mood of the scene." if detect_mood else "Skip." }

                **STORYTELLING:**
                { f"Write a {story_mode} based on the visual elements." if story_mode != "Off" else "Skip." }

                **QUALITY ANALYSIS:**
                { "Assess quality." if detect_quality else "Skip." }

                **SOCIAL MEDIA:**
                { "Provide #hashtags and crop sizes." if (add_hashtags or crop_suggest) else "Skip." }

                **USER Q&A:**
                { f"Answer: {user_qa}" if user_qa else "No question." }

                **COMPARISON:**
                { "Compare the images." if enable_comparison else "Not requested." }
                """

                response = gemini_model.generate_content([prompt] + img_input)
                
                st.markdown("---")
                st.subheader("📝 Results")
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
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if chat_user_input := st.chat_input("Ask me anything..."):
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
