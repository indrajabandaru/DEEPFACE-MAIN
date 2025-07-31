import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import pyttsx3
import time
import os
import math
from PIL import Image

st.set_page_config(page_title="Emotion Insight", layout="wide", initial_sidebar_state="collapsed")

st.markdown("<h1 style='text-align:center; color:#00f5d4;'>🎭 Emotion Insight - AI Powered Detector</h1>", unsafe_allow_html=True)

COLOR_MAP = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (255, 255, 0),
    "neutral": (200, 200, 200),
    "fear": (255, 0, 255),
    "disgust": (0, 255, 255)
}

engine = pyttsx3.init()
engine.setProperty('rate', 160)
last_spoken = None

if 'run' not in st.session_state:
    st.session_state.run = False
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
if 'emotion_images' not in st.session_state:
    st.session_state.emotion_images = []

col1, col2 = st.columns(2)
if col1.button("▶ Start Webcam"):
    st.session_state.run = True
if col2.button("⏹ Stop Webcam"):
    st.session_state.run = False

frame_window = st.image([])
chart_placeholder = st.empty()
snapshot_btn = st.button("📷 Capture Snapshot")

if st.session_state.run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("❌ Webcam not accessible")
    else:
        frame_count = 0
        last_result = None

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam.")
                break

            frame_count += 1
            h, w, _ = frame.shape
            small_frame = cv2.resize(frame, (320, 240))

            try:
                if frame_count % 5 == 0:
                    result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)
                    last_result = result[0]
                    dominant_emotion = last_result['dominant_emotion']
                    confidence = last_result['emotion'][dominant_emotion]

                    # Override low confidence sad -> happy (optional fix)
                    if dominant_emotion == "sad" and confidence < 60:
                        dominant_emotion = "happy"

                    if dominant_emotion != last_spoken:
                        engine.say(f"You look {dominant_emotion}")
                        engine.runAndWait()
                        last_spoken = dominant_emotion

                    timestamp = time.strftime("%H:%M:%S")
                    st.session_state.emotion_log.append((timestamp, dominant_emotion))

                    face_img = cv2.resize(frame, (100, 100))
                    face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    st.session_state.emotion_images.append(face_img_pil)
                    if len(st.session_state.emotion_images) > 5:
                        st.session_state.emotion_images.pop(0)

                if last_result:
                    dominant_emotion = last_result['dominant_emotion']
                    confidence = last_result['emotion'][dominant_emotion]
                    color = COLOR_MAP.get(dominant_emotion, (255, 255, 255))

                    label = f"{dominant_emotion.upper()} ({confidence:.1f}%)"
                    y_pos = int(60 + 20 * math.sin(time.time() * 2))

                    cv2.rectangle(frame, (30, 20), (500, 100), (0, 0, 0), -1)
                    cv2.putText(frame, label, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

                    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), color, 4)

                    bar_length = int(confidence * 4)
                    cv2.rectangle(frame, (30, h - 30), (30 + bar_length, h - 10), color, -1)
                    cv2.putText(frame, "Confidence", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    emotion_df = pd.DataFrame(st.session_state.emotion_log, columns=["Time", "Emotion"])
                    count_df = emotion_df['Emotion'].value_counts().reset_index()
                    count_df.columns = ['Emotion', 'Count']
                    chart_placeholder.bar_chart(data=count_df.set_index('Emotion'))

                    if snapshot_btn:
                        os.makedirs("snapshots", exist_ok=True)
                        filename = f"snapshots/snapshot_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        st.success(f"📸 Snapshot saved: {filename}")

            except Exception as e:
                st.warning(f"⚠ {e}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb)

        cap.release()

st.subheader("🖼 Recent Emotion Faces")
if st.session_state.emotion_images:
    st.image(st.session_state.emotion_images, width=100)

st.subheader("🕒 Last 10 Emotions Timeline")
if st.session_state.emotion_log:
    timeline_df = pd.DataFrame(st.session_state.emotion_log[-10:], columns=["Time", "Emotion"])
    st.table(timeline_df[::-1])

if st.session_state.emotion_log:
    df_log = pd.DataFrame(st.session_state.emotion_log, columns=["Time", "Emotion"])
    csv = df_log.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Emotion Log", csv, "emotion_log.csv", "text/csv")