import os
import time
import cv2
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv()
SESSION_ID = os.getenv("SESSION_ID", "demo_session")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Download YOLOv8 face detection model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="keremberke/yolov8n-face",
    filename="yolov8n-face.onnx",
    token=HUGGINGFACE_TOKEN
)

# Load the ONNX model using OpenCV DNN
net = cv2.dnn.readNetFromONNX(model_path)

# Streamlit App UI
st.title("🛡️ Online Proctoring System (Image-based)")
uploaded_file = st.file_uploader("Upload image frame for proctoring (jpg/png)", type=["jpg", "jpeg", "png"])
log = []

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    h, w = image.shape[:2]

    # Prepare blob and run inference
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for detection in detections[0]:
        confidence = detection[4]
        if confidence > 0.5:
            x, y, bw, bh = detection[0] * w, detection[1] * h, detection[2] * w, detection[3] * h
            x1 = int(x - bw / 2)
            y1 = int(y - bh / 2)
            x2 = int(x + bw / 2)
            y2 = int(y + bh / 2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Decision logic
    if len(faces) == 0:
        status = "⚠️ No face detected"
    elif len(faces) > 1:
        status = "⚠️ Multiple faces detected"
    else:
        status = "✅ One face detected"

    log.append({
        "timestamp": time.time(),
        "status": status,
        "session_id": SESSION_ID
    })

    # Show results
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Status: {status}")
    st.text(f"Detection Result: {status}")

    # Export button
    if st.button("📤 Export Log"):
        with open(f"log_{SESSION_ID}.json", "w") as f:
            json.dump(log, f, indent=4)
        st.success("Log exported successfully!")
