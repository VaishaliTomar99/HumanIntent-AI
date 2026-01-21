import cv2
import torch
import numpy as np
from torchvision import models, transforms
from train_model import CNN_LSTM      # reuse same class
from collections import deque

# --------- Settings ----------
MODEL_PATH = "models/cnn_lstm_best.pth"
NUM_FRAMES = 16
FRAME_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Load model ----------
model = CNN_LSTM(hidden_size=256, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------- Transform ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor()
])

# --------- Frame buffer ----------
frames_buffer = deque(maxlen=NUM_FRAMES)

# --------- Open webcam ----------
cap = cv2.VideoCapture(0)   # or replace 0 with your CCTV stream URL

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Real-time Violence Detection running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # preprocess frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))
    frames_buffer.append(rgb)

    # Only predict when we have enough frames
    if len(frames_buffer) == NUM_FRAMES:
        # prepare tensor (1, T, C, H, W)
        frames = [transform(f) for f in list(frames_buffer)]
        clip = torch.stack(frames).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(clip)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

        label = "VIOLENT" if pred == 1 else "SAFE"
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)

        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Real-Time Violence Detection", frame)

    # quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting...")
