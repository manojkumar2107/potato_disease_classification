from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "potato_model_v1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (keep the same order used during training)
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "Please select an image"

    # Read uploaded image
    img = Image.open(file.stream).convert("RGB")

    # Resize image
    img = img.resize((256, 256))

    # Convert image to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])
    confidence = round(float(np.max(prediction[0])) * 100, 2)

    predicted_class = CLASS_NAMES[predicted_index]

    return f"""
        <h1>Prediction Result</h1>
        <h2>{predicted_class}</h2>
        <p>Confidence: {confidence}%</p>
        <a href="/">Try another image</a>
    """


if __name__ == "__main__":
    app.run(debug=True)