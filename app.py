import os
import cv2
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv()
SESSION_ID = os.getenv("SESSION_ID", "unknown")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Download model from Hugging Face Hub (example: yolov8 face detector)
model_path = hf_hub_download(repo_id="keremberke/yolov8n-face", filename="yolov8n-face.onnx", token=HUGGINGFACE_TOKEN)

# Load ONNX model
net = cv2.dnn.readNetFromONNX(model_path)

st.title("🤖 Online Proctoring System with Hugging Face Model")
run = st.checkbox('Start Proctoring')

log = []
no_face_counter = 0
max_no_face_duration = 10  # seconds

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        faces = []
        h, w = frame.shape[:2]

        for detection in detections[0]:
            confidence = detection[4]
            if confidence > 0.5:
                x, y, bw, bh = detection[0] * w, detection[1] * h, detection[2] * w, detection[3] * h
                x1 = int(x - bw / 2)
                y1 = int(y - bh / 2)
                x2 = int(x + bw / 2)
                y2 = int(y + bh / 2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if len(faces) == 0:
            no_face_counter += 1
            status = "⚠️ No face detected"
        elif len(faces) > 1:
            status = "⚠️ Multiple faces detected"
        else:
            no_face_counter = 0
            status = "✅ Face detected"

        if status != "✅ Face detected":
            log.append({
                "timestamp": time.time(),
                "status": status,
                "session_id": SESSION_ID
            })

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.text(f"Status: {status}")

        if no_face_counter * (1/30) > max_no_face_duration:
            st.warning("No face detected for too long. Stopping session.")
            break

        if not run:
            break

    cap.release()
    st.success("Proctoring session ended")

    if st.button("Export Log"):
        import json
        with open(f"log_{SESSION_ID}.json", "w") as f:
            json.dump(log, f, indent=4)
        st.success("Log exported successfully")
