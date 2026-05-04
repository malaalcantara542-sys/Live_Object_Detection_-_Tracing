import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os

# Cache the model so it doesn't reload every rerun
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Folder for saving frames
SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Sidebar options
st.sidebar.header("Enhancement Options")
enable_counting = st.sidebar.checkbox("Enable Object Counting", True)
enable_alerts = st.sidebar.checkbox("Enable Alerts", True)
enable_saving = st.sidebar.checkbox("Save Detected Frames", True)

alert_object = st.sidebar.text_input("Alert for object:", "cell phone")

# Video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # Extract detections
    boxes = results[0].boxes
    cls = boxes.cls if boxes is not None else []

    # Convert class IDs to labels
    labels = [model.names[int(c)] for c in cls]

    # -----------------------------
    # 1. OBJECT COUNTING
    # -----------------------------
    if enable_counting:
        person_count = labels.count("person")
        st.sidebar.write(f"👥 People detected: **{person_count}**")

    # -----------------------------
    # 2. TRIGGER ALERTS
    # -----------------------------
    if enable_alerts:
        if alert_object.lower() in [l.lower() for l in labels]:
            st.warning(f"⚠️ ALERT: {alert_object.upper()} detected!")

    # -----------------------------
    # 3. SAVE DETECTED FRAMES
    # -----------------------------
    if enable_saving and len(labels) > 0:
        timestamp = int(time.time() * 1000)
        filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
        cv2.imwrite(filename, img)

    # Annotate frame
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Start WebRTC streamer
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,  # smoother performance
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
