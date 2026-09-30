%%writefile app.py
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Object Detection & Tracking", page_icon="📹", layout="centered")

st.title("📹 Real-Time Object Detection & Tracking")
st.write("Upload an image to detect objects and display visual bounding boxes using YOLOv8.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect & Track Objects", type="primary"):
        img_array = np.array(image)
        results = model.track(img_array, persist=True)
        annotated_frame = results[0].plot()
        
        st.subheader("Detection Output:")
        st.image(annotated_frame, caption="Processed Image with Bounding Boxes", use_column_width=True)
