# Run the app bu doing: streamlit run streamlit_app.py

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# 1. Setup paths so Streamlit can find your 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.model import get_model

# 2. Load the model directly (No API needed!)
@st.cache_resource # This keeps the model in memory so it stays fast
def load_my_model():
    model_path = os.path.join(root_dir, "models", "best_model.pth")
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_my_model()

# 3. Handle the Image Upload
st.title("🍎 QualityLens")
uploaded_file = st.file_uploader("Upload a fruit photo...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Target Image", use_container_width=True)
    
    if st.button("Predict Quality"):
        # Preprocess exactly like you did in training
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            prob = F.softmax(output, dim=1)
            conf, idx = torch.max(prob, 1)

        labels = ["Fresh", "Rotten"]
        st.write(f"### Prediction: {labels[idx.item()]}")
        st.write(f"Confidence: {conf.item()*100:.2f}%")