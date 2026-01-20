# 🔍 QualityLens

QualityLens is a computer vision application designed to classify fruit and evaluate its physical quality. By combining a **custom-trained Convolutional Neural Network (CNN)** with advanced image preprocessing, the app provides both species identification and a "Quality Score" percentage to help determine freshness and integrity.

## 💻 Technologies

* **Python**
* **PyTorch** (Custom CNN Architecture)
* **FastAPI** (Backend API)
* **Streamlit** (Frontend UI)
* **OpenCV** (Image Preprocessing)
* **NumPy / Matplotlib** (Data Analysis)

## ✨ Features

* **Multi-Class Classification:** Specifically trained to identify **Apples**, **Oranges**, and **Bananas**.
* **Quality Scoring:** Provides a percentage-based score (0-100%) evaluating the quality and freshness of the fruit.
* **Custom Inference Engine:** Uses a custom-built CNN architecture rather than a generic pre-trained model.
* **Pre-processing Pipeline:** Leverages OpenCV to normalize lighting and scale images for high-accuracy inference.

## 📁 The Process

To build QualityLens, I utilized the **Kaggle Fruits Dataset** to train a deep learning model from scratch. Unlike standard transfer learning, I designed a **Custom Convolutional Neural Network (CNN)** in PyTorch to understand specific spatial features of fruit textures, shapes, and surface defects.



[Image of a convolutional neural network architecture]


The workflow is split into three main stages:

1.  **Data & Training:** Images were preprocessed using **OpenCV** for consistency (resizing, grayscale conversion, and normalization). The model was trained using PyTorch, optimizing for both classification accuracy and quality regression.
2.  **API Layer:** A **FastAPI** server handles the model inference. It receives image data via POST requests, runs it through the saved `.pth` model weights, and returns a JSON response containing the label and the calculated quality percentage.
3.  **User Interface:** A **Streamlit** dashboard allows users to upload images and visualize the results instantly with clean progress bars and confidence levels.

### 🚦 Running the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mickman7/QualityLens.git](https://github.com/Mickman7/QualityLens.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the FastAPI Backend:**
    ```bash
    uvicorn backend.main:app --reload
    ```

4.  **Run the Streamlit Frontend:**
    ```bash
    streamlit run frontend/app.py
    ```

5.  **Analyze:** Open your browser to the local Streamlit URL (usually `http://localhost:8501`) and upload a `.jpg` or `.png` of a fruit.

## 📺 Preview

*(Replace this with a screenshot of your Streamlit dashboard showing the classification and quality score)*

---
