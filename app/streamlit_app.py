# Run the app bu doing: streamlit run streamlit_app.py

import streamlit as st
import torch
import os
import sys
from PIL import Image
import torch.nn.functional as F 
from torchvision import transforms

# --- 1. SETUP PATHS SO PYTHON SEES 'src' ---
# Get the directory where this script is (QualityLens/app)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (QualityLens)
root_dir = os.path.join(current_dir, "..")
# Add root to sys.path
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 2. NOW IMPORT YOUR MODEL ---
# This must come AFTER the sys.path.append
from src.model import get_model 

# --- 3. LOAD THE MODEL BRAIN ---
# Build the absolute path to the .pth file
model_path = os.path.join(root_dir, "models", "best_model.pth")

# Create the model structure and fill it with your saved weights
model = get_model(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

st.success("Model loaded successfully!")


st.title("QualityLens: Fruit Freshness Detector")
st.write("Upload a photo of a fruit to check its quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Check Quality'):
        # 1. Prepare the image (Preprocessing)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # 2. Run the model (Inference)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, index = torch.max(probabilities, 1)

        # 3. Show the results
        classes = ["Fresh", "Rotten"]
        prediction = classes[index.item()]
        
        if prediction == "Fresh":
            st.success(f"Result: {prediction} (Confidence: {confidence.item()*100:.2f}%)")
        else:
            st.error(f"Result: {prediction} (Confidence: {confidence.item()*100:.2f}%)")