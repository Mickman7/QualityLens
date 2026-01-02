from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn.functional as F
from PIL import Image
import io
import sys
import os

# Import your model structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import get_model

app = FastAPI()

# Load the brain
device = torch.device("cpu") # Use CPU for the API for stability
model = get_model(num_classes=2)
model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Preprocess (Must match what you did in training!)
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # 3. Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, index = torch.max(probabilities, 1)

    classes = ["Fresh", "Rotten"]
    return {
        "prediction": classes[index.item()],
        "confidence": f"{confidence.item() * 100:.2f}%"
    }