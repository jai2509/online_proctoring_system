import cv2
import time
import numpy as np
import streamlit as st
import json
import os
from ultralytics import YOLO

# Streamlit Page Config
st.set_page_config(page_title="🛡️ Online Proctoring System", layout="centered")
st.title("🛡️ Online Proctoring System (YOLOv8 + Face + Flash)")

# Proctoring Settings
run = st.checkbox("✅ Start Proctoring")
log = []
screenshot_dir = "logs"
os.makedirs(screenshot_dir, exist_ok=True)

no_face_counter = 0
max_no_face_duration = 10  # seconds
fps = 10
start_time = None

# Load YOLOv8 model from Ultralytics hub (no need for local .pt file)
model = YOLO('yolov8n')  # Automatically downloads weights if needed

# Load Haar face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Warning tracking
warnings_count = {
    "no_face": 0,
    "multiple_faces": 0,
    "flash": 0,
    "phone": 0
}

# Utility Functions
def take_screenshot(frame, reason):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(screenshot_dir, f"{reason}_{timestamp}.jpg")
    cv2.imwrite(path, frame)

def detect_flashlight(frame):
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return brightness > 220  # threshold for flash

def detect_phone_yolo(frame):
    results = model(frame, verbose=False)[0]
    for r in results.boxes.data:
        cls = int(r[5])
        label = model.names[cls]
        if label.lower() in ["cell phone", "mobile phone", "phone"]:
            return True
    return False

# Main Logic
if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    status_placeholder = st.empty()
    dashboard = st.empty()
    warning_placeholder = st.empty()
    start_time = time.time()

    if not cap.isOpened():
        st.error("❌ Webcam not accessible")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam.")
                break

            elapsed = int(time.time() - start_time)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw face box
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Detection logic
            flash = detect_flashlight(frame)
            phone = detect_phone_yolo(frame)

            if flash:
                status = "⚡ Flashlight Detected"
                warning_placeholder.error(status)
                take_screenshot(frame, "flash")
                warnings_count["flash"] += 1

            elif phone:
                status = "📱 Phone Detected"
                warning_placeholder.error(status)
                take_screenshot(frame, "phone")
                warnings_count["phone"] += 1

            elif len(faces) == 0:
                no_face_counter += 1
                status = "⚠️ No Face Detected"
                warning_placeholder.warning(status)
                take_screenshot(frame, "no_face")
                warnings_count["no_face"] += 1

            elif len(faces) > 1:
                status = "🚨 Multiple Faces Detected"
                warning_placeholder.error(status)
                take_screenshot(frame, "multiple_faces")
                warnings_count["multiple_faces"] += 1

            else:
                status = "✅ Face Detected"
                no_face_counter = 0
                warning_placeholder.empty()

            # Log
            log.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status
            })

            # Dashboard
            dashboard.info(f"""
            ⏱️ **Time Elapsed:** {elapsed} sec  
            ❌ No Face: {warnings_count['no_face']}  
            👥 Multiple Faces: {warnings_count['multiple_faces']}  
            📱 Phone Detections: {warnings_count['phone']}  
            ⚡ Flashlight Events: {warnings_count['flash']}
            """)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            status_placeholder.markdown(f"### Status: {status}")

            if no_face_counter * (1 / fps) > max_no_face_duration:
                st.error("🛑 No face detected for too long. Session ended.")
                break

            time.sleep(1 / fps)

        cap.release()
        st.success("✅ Proctoring session ended.")

        st.download_button(
            label="📥 Download Proctoring Log",
            data=json.dumps(log, indent=2),
            file_name="proctoring_log.json",
            mime="application/json"
        )
