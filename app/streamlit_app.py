# # Run the app bu doing: streamlit run streamlit_app.py

# import streamlit as st
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import os
# import sys

# # 1. Setup paths so Streamlit can find your 'src' folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.join(current_dir, "..")
# if root_dir not in sys.path:
#     sys.path.append(root_dir)

# from src.model import get_model

# # 2. Load the model directly (No API needed!)
# @st.cache_resource # This keeps the model in memory so it stays fast
# def load_my_model():
#     model_path = os.path.join(root_dir, "models", "best_model.pth")
#     model = get_model(num_classes=2)
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     return model

# model = load_my_model()

# # 3. Handle the Image Upload
# st.title("QualityLens")
# uploaded_file = st.file_uploader("Upload a fruit photo...", type=["jpg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Target Image", use_container_width=True)
    
#     if st.button("Predict Quality"):
#         # Preprocess exactly like you did in training
#         preprocess = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         input_tensor = preprocess(image).unsqueeze(0)

#         # Predict
#         with torch.no_grad():
#             output = model(input_tensor)
#             prob = F.softmax(output, dim=1)
#             conf, idx = torch.max(prob, 1)

#         labels = ["Fresh", "Rotten"]
#         st.write(f"### Prediction: {labels[idx.item()]}")
#         st.write(f"Confidence: {conf.item()*100:.2f}%")





import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# --- 1. SETUP & PATHING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.model import get_model

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def change_page(page_name):
    st.session_state.page = page_name
    st.rerun()

@st.cache_resource
def load_my_model():
    model_path = os.path.join(root_dir, "models", "best_model.pth")
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_my_model()

# --- 2. ENHANCED CSS FOR SPACING ---
st.markdown("""
    <style>
    /* Force wide layout and white background */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    
    /* Hero Typography */
    .hero-title { 
        font-size: 72px !important; 
        font-weight: 800; 
        line-height: 1.0; 
        color: #000; 
        margin-bottom: 1.5rem;
    }
    .hero-sub { 
        font-size: 20px; 
        color: #444; 
        margin-bottom: 2.5rem; 
        line-height: 1.6;
    }
    
    /* Process Section Spacing */
    .process-container { margin-top: 100px; margin-bottom: 50px; }
    .process-header { font-size: 42px; font-weight: 700; text-align: center; margin-bottom: 15px; }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #000 !important;
        color: #fff !important;
        border-radius: 2px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        border: none;
    }
    
    /* Image Grid Styling */
    .img-card { border-radius: 8px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE HOME PAGE ---
if st.session_state.page == 'Home':
    # Navigation Header
    header_left, header_right = st.columns([1, 1])
    with header_left:
        st.write("### QualityLens")
    

    # Hero Section
    hero_col_left, hero_col_right = st.columns([1.5, 1], gap="large")
    
    with hero_col_left:
        st.markdown('<h1 class="hero-title">Know your fruit at a glance</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Upload an image and our machine learning model instantly tells you if your fruit is fresh or rotten. Simple, fast, and accurate every time.</p>', unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns([1, 2])
        with btn_col1:
            if st.button("Upload"):
                change_page('Upload')
        with btn_col2:
            st.button("Learn more")

    with hero_col_right:
            st.image("https://images.unsplash.com/photo-1610832958506-aa56368176cf?auto=format&fit=crop&w=800&q=80", use_container_width=True)


    # Process Section
    st.container()
    st.write("---")
    st.markdown('<p style="text-align:center; color:#888; text-transform:uppercase; letter-spacing:2px; font-size:14px; margin-top:80px;">Process</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="process-header">Three simple steps to results</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#666; margin-bottom:60px;">The entire process takes less than a minute from upload to classification.</p>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.markdown("### **Upload your image**")
        st.write("Select a photo of your fruit from your device.")
    with p2:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.markdown("### **Our model analyzes**")
        st.write("Advanced algorithms examine texture, color, and condition.")
    with p3:
        st.image("https://images.unsplash.com/photo-1490818387583-1baba5e638af?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.markdown("### **Get your result**")
        st.write("Receive a clear fresh or rotten classification instantly.")

    # --- 5. FEATURES SECTION (New) ---
    st.container()
    st.write("---") # Visual separator
    st.write("")
    st.write("")

    # Main container for the Features section
    feat_left, feat_right = st.columns([1.2, 1], gap="large")

    with feat_left:
        st.markdown('<p style="color:gray; text-transform:uppercase; letter-spacing:2px; font-size:14px;">Features</p>', unsafe_allow_html=True)
        st.markdown('<h1 style="font-size: 56px; font-weight: 800; line-height: 1.1; margin-bottom: 30px;">Everything you need to classify fruit</h1>', unsafe_allow_html=True)
        
        # Buttons aligned horizontally
        f_btn1, f_btn2 = st.columns([1, 2])
        with f_btn1:
            if st.button("Start now", key="feat_start"):
                change_page('Upload')
        with f_btn2:
            st.markdown('<p style="margin-top: 12px; font-weight: 600; cursor: pointer;">Arrow ❯</p>', unsafe_allow_html=True)

    with feat_right:
        # Feature list data
        features = [
            {"icon": "🖼️", "title": "Instant classification", "text": "Get results immediately after uploading your image."},
            {"icon": "🎯", "title": "Accurate detection", "text": "Machine learning trained on thousands of fruit samples."},
            {"icon": "🔒", "title": "Your privacy matters", "text": "Images are processed securely and never stored."},
            {"icon": "⚡", "title": "Lightning-fast results", "text": "Upload and know within seconds whether your fruit is at its peak or past it."}
        ]

        for f in features:
            # Create a mini-column for the icon vs text to mimic the vertical line look
            icon_col, text_col = st.columns([1, 5])
            with icon_col:
                st.markdown(f"### {f['icon']}")
                # This creates the visual "vertical line" effect between icons
                st.markdown('<div style="border-left: 2px solid #eee; height: 50px; margin-left: 15px;"></div>', unsafe_allow_html=True)
            with text_col:
                st.markdown(f"#### **{f['title']}**")
                st.markdown(f'<p style="color: #666; margin-top: -10px;">{f["text"]}</p>', unsafe_allow_html=True)
            st.write("") # Padding between items

    # --- 6. FOOTER SECTION ---
    st.write("")
    st.write("")
    st.write("")
    st.divider()

    # Top row of footer: Logo, Links, and Socials
    foot_col1, foot_col2, foot_col3 = st.columns([1, 2, 1])

    with foot_col1:
        st.write("### QualityLens")

    with foot_col2:
        # Mimicking the centered horizontal menu
        st.markdown("""
            <div style="display: flex; justify-content: center; gap: 25px; color: #000; font-weight: 500; margin-top: 10px;">
                <span style="cursor: pointer;">Upload</span>
                <span style="cursor: pointer;">Results</span>
                <span style="cursor: pointer;">FAQ</span>
                <span style="cursor: pointer;">Learn</span>
                <span style="cursor: pointer;">Company</span>
            </div>
        """, unsafe_allow_html=True)

    with foot_col3:
        # Social Media Icons using Emoji/Markdown (matches your screenshot)
        st.markdown("""
            <div style="display: flex; justify-content: flex-end; gap: 15px; font-size: 20px; margin-top: 5px;">
                <span>🌐</span> <span>📸</span> <span>🐦</span> <span>🔗</span> <span>🎥</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("") # Padding

    # Bottom row of footer: Copyright and Legal
    copy_col1, copy_col2 = st.columns([1, 1])
    with copy_col1:
        st.markdown('<p style="color: #888; font-size: 14px;">© 2025 QualityLens Platform</p>', unsafe_allow_html=True)

    with copy_col2:
        st.markdown("""
            <div style="display: flex; justify-content: flex-end; gap: 20px; color: #888; font-size: 14px;">
                <span style="text-decoration: underline; cursor: pointer;">Privacy policy</span>
                <span style="text-decoration: underline; cursor: pointer;">Terms of service</span>
                <span style="text-decoration: underline; cursor: pointer;">Cookie settings</span>
            </div>
        """, unsafe_allow_html=True)

# --- 4. THE UPLOAD PAGE ---
else:
    if st.button("Back"):
        change_page('Home')
    
    st.title("Upload & Predict")
    uploaded_file = st.file_uploader("Upload a fruit photo...", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Target Image", width=300) # Fixed width prevents squashing
        
        if st.button("Predict Quality"):
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                prob = F.softmax(output, dim=1)
                conf, idx = torch.max(prob, 1)

            labels = ["Fresh", "Rotten"]
            res = labels[idx.item()]
            if res == "Fresh":
                st.success(f"Prediction: {res} ({conf.item()*100:.2f}%)")
            else:
                st.error(f"Prediction: {res} ({conf.item()*100:.2f}%)")