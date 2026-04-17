from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "railway_defect_binary_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    file_path = "temp.jpg"
    file.save(file_path)

    img = preprocess_image(file_path)
    prediction = model.predict(img)[0][0]

    os.remove(file_path)

    return jsonify({
        "defect": bool(prediction > 0.5),
        "confidence": float(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
