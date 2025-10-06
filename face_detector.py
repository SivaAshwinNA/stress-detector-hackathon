import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

st.set_page_config(page_title="Live Stress Detector", layout="wide")

st.title("🧠 Real-Time Stress Detection System")
st.markdown("Detects stress levels from your facial expressions in real-time using DeepFace.")

# Start webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("No camera input detected.")
        break

    # Flip horizontally for natural selfie view
    frame = cv2.flip(frame, 1)

    try:
        # Analyze the frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Convert to "stress level" conceptually
        stress_map = {
            'happy': 10,
            'neutral': 30,
            'sad': 60,
            'fear': 75,
            'angry': 85,
            'disgust': 70,
            'surprise': 40
        }
        stress_level = stress_map.get(emotion, 50)

        # Display text overlay
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Stress Level: {stress_level}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Color code border
        color = (0,255,0) if stress_level < 40 else (0,255,255) if stress_level < 70 else (0,0,255)
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), color, 10)

    except Exception as e:
        print("Detection failed:", e)
    
    # Convert color (OpenCV → RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

camera.release()
