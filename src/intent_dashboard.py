import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from torchvision import transforms
from collections import deque
from train_model import CNN_LSTM

# ---------------- CONFIG ----------------
MODEL_PATH = "models/cnn_lstm_best.pth"
DATASET_PATH = "data/A-Dataset-for-Automatic-Violence-Detection-in-Videos/violence-detection-dataset"
FRAME_SIZE = 112
NUM_FRAMES = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = CNN_LSTM(hidden_size=256, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor()
])

# ---------------- LOAD CSV LABELS ----------------
violent_csv = pd.read_csv(os.path.join(DATASET_PATH, "violent-action-classes.csv"), sep=";")
nonviolent_csv = pd.read_csv(os.path.join(DATASET_PATH, "nonviolent-action-classes.csv"), sep=";")

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Real-Time Intention Prediction Dashboard", layout="wide")
st.title("🧠 Real-Time Human Intention & Violence Detection Dashboard")

mode = st.sidebar.radio("Select Mode", ["📁 Dataset Video", "🎥 Live Webcam"])

# =========================================================================================
# 📁 MODE 1: DATASET VIDEO
# =========================================================================================
if mode == "📁 Dataset Video":
    st.header("🎬 Dataset Video Prediction")

    category = st.sidebar.radio("Choose Category", ["violent", "non-violent"])

    video_dir_cam1 = os.path.join(DATASET_PATH, category, "cam1")
    video_dir_cam2 = os.path.join(DATASET_PATH, category, "cam2")

    all_videos = sorted(
        [os.path.join(video_dir_cam1, f) for f in os.listdir(video_dir_cam1) if f.endswith(".mp4")] +
        [os.path.join(video_dir_cam2, f) for f in os.listdir(video_dir_cam2) if f.endswith(".mp4")]
    )

    video_choice = st.sidebar.selectbox("🎥 Select Video", all_videos)

    filename = os.path.basename(video_choice)
    if category == "violent":
        row = violent_csv[violent_csv["FILE"] == filename]
    else:
        row = nonviolent_csv[nonviolent_csv["FILE"] == filename]
    ground_truth = row[" ACTION CLASSES"].values[0] if len(row) > 0 else "Unknown"

    st.markdown(f"**Ground Truth Action(s):** `{ground_truth}`")

    cap = cv2.VideoCapture(video_choice)
    frames_buffer = deque(maxlen=NUM_FRAMES)
    preds = []

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))
        frames_buffer.append(resized)

        if len(frames_buffer) == NUM_FRAMES:
            frames = [transform(f) for f in list(frames_buffer)]
            clip = torch.stack(frames).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1)[0]
                pred = torch.argmax(probs).item()
                preds.append(pred)

            label = "VIOLENT" if pred == 1 else "SAFE"
            color = (255, 0, 0) if pred == 1 else (0, 255, 0)
            cv2.putText(frame, f"{label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # ✅ fixed deprecation: use_container_width instead of use_column_width
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                      channels="RGB",
                      use_container_width=True)

    cap.release()

    if len(preds) > 0:
        final_pred = int(np.round(np.mean(preds)))
        st.success(f"✅ **Model Prediction (Overall):** {'VIOLENT' if final_pred == 1 else 'SAFE'}")
    else:
        st.warning("⚠️ Not enough frames for prediction.")

# =========================================================================================
# 🎥 MODE 2: LIVE WEBCAM
# =========================================================================================
else:
    st.header("🎥 Live Webcam Intention Prediction")

    run = st.checkbox("Start Webcam")
    stframe = st.empty()
    frames_buffer = deque(maxlen=NUM_FRAMES)

    cap = cv2.VideoCapture(0)
    st.markdown("Press the checkbox to start or stop webcam.")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))
        frames_buffer.append(resized)

        if len(frames_buffer) == NUM_FRAMES:
            frames = [transform(f) for f in list(frames_buffer)]
            clip = torch.stack(frames).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1)[0]
                pred = torch.argmax(probs).item()

            label = "VIOLENT" if pred == 1 else "SAFE"
            color = (255, 0, 0) if pred == 1 else (0, 255, 0)
            cv2.putText(frame, f"{label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # ✅ fixed deprecation + auto-resize
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                      channels="RGB",
                      use_container_width=True)

    cap.release()
