# Run the app bu doing: streamlit run streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import torch
from src.model import get_model

model = get_model(num_classes=2)
model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
model.eval()

st.title("QualityLens: Fruit Freshness Detector")
st.write("Upload a photo of a fruit to check its quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # When user clicks the button
    if st.button('Check Quality'):
        # Send to API
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Result: {result['prediction']}")
            st.info(f"Confidence: {result['confidence']}")
        else:
            st.error("API Error. Make sure the FastAPI server is running!")