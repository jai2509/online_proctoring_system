import cv2
import time
import numpy as np
import streamlit as st

st.title("🛡️ Online Proctoring System (OpenCV)")

run = st.checkbox("Start Proctoring")
log = []
no_face_counter = 0
max_no_face_duration = 10  # seconds
fps = 10

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Webcam not accessible")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Status Logic
        if len(faces) == 0:
            no_face_counter += 1
            status = "⚠️ No face detected"
        elif len(faces) > 1:
            status = "⚠️ Multiple faces detected"
        else:
            no_face_counter = 0
            status = "✅ Face detected"

        log.append({
            "timestamp": time.time(),
            "status": status
        })

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        st.text(f"Status: {status}")

        if no_face_counter * (1/fps) > max_no_face_duration:
            st.warning("🛑 No face detected for too long. Ending session.")
            break

        time.sleep(1/fps)

    cap.release()
    st.success("✅ Proctoring session ended.")

    if st.button("Download Log"):
        import json
        with open("proctoring_log.json", "w") as f:
            json.dump(log, f, indent=2)
        st.success("Log saved as `proctoring_log.json`")
