import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="ID Ribbon Detection",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 ID Ribbon Detection")

# Load Model
MODEL_PATH = "runs_detect_id_card_yolo11m_75epochs_weights_best.pt"

@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(model_path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Upload Image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    input_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(input_img)
    
    # Run detection
    with st.spinner("Detecting..."):
        results = model.predict(img_np, verbose=False)
    
    # Get annotated image
    result = results[0]
    annotated = result.plot()
    annotated_pil = Image.fromarray(annotated)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_img, use_container_width=True)
    
    with col2:
        st.subheader("Detection Result")
        st.image(annotated_pil, use_container_width=True)
