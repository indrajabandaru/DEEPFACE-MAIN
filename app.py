import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

st.title("Live Emotion Detection")

# Initialize session state to control loop
if 'run' not in st.session_state:
    st.session_state.run = False

# Start button
if st.button("Start Webcam"):
    st.session_state.run = True

# Stop button
if st.button("Stop Webcam", key="stop_button"):
    st.session_state.run = False

frame_window = st.image([])  # placeholder for video frames

# Run video feed loop if run flag is True
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible")
    else:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][dominant_emotion]

                # Draw emotion label
                cv2.putText(frame, f"{dominant_emotion} ({confidence:.1f}%)",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                st.write(f"Error: {e}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb)

        cap.release()