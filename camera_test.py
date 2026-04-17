import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# 🔹 Load trained model
model = load_model("railway_defect_binary_model.h5")

# 🔹 Load video file
cap = cv2.VideoCapture("WhatsApp Video 2026-04-17 at 7.34.05 PM.mp4")

# 🔹 Check video
if not cap.isOpened():
    print("❌ Video not opened")
    exit()
else:
    print("✅ Video loaded")

# 🔹 Create folder for crack images
os.makedirs("crack_frames", exist_ok=True)

# 🔹 Preprocess function
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    return img

frame_count = 0
saved_count = 0

# 🔥 Adjustable threshold (IMPORTANT)
THRESHOLD = 0.3   # change between 0.2–0.5

while True:
    ret, frame = cap.read()

    if not ret:
        print("\n✅ Video completed")
        break

    frame_count += 1

    # 🔹 Skip frames for speed
    if frame_count % 5 != 0:
        continue

    # 🔹 Preprocess
    img = preprocess(frame)

    # 🔹 Predict
    prediction = model.predict(img, verbose=0)
    value = prediction[0][0]

    print(f"Frame {frame_count} → Prediction: {value:.3f}")

    # 🔴 Save only crack frames
    if value > THRESHOLD:
        saved_count += 1

        filename = f"crack_frames/crack_{saved_count}.jpg"
        cv2.imwrite(filename, frame)

        print(f"❌ Crack detected → saved: {filename}")

cap.release()

print(f"\n🔥 Total crack frames saved: {saved_count}")