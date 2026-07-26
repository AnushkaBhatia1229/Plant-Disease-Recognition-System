from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import os
import tensorflow as tf
import gdown

app = Flask(__name__)

# ============================
# Download AI Model (Only Once)
# ============================

MODEL_PATH = "plant_disease_recog_model_pwp.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading AI model from Google Drive...")
    gdown.download(
        id="14BUv71OaCFl4_b0ypL1rdob-zM_Cuwjd",
        output=MODEL_PATH,
        quiet=False
    )

print("Loading AI model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model Loaded Successfully!")

# ============================
# Load Disease Labels
# ============================

with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# ============================
# Upload Folder
# ============================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================
# Routes
# ============================

@app.route("/uploads/<path:filename>")
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")
def home():
    return render_template("home.html")

# ============================
# Image Processing
# ============================

def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img, verbose=0)
    prediction_label = plant_disease[np.argmax(prediction)]
    return prediction_label

# ============================
# Upload Image
# ============================

@app.route("/upload/", methods=["POST"])
def uploadimage():

    if "img" not in request.files:
        return redirect("/")

    image = request.files["img"]

    if image.filename == "":
        return redirect("/")

    filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    image.save(filepath)

    prediction = model_predict(filepath)

    return render_template(
        "home.html",
        result=True,
        imagepath=f"/uploads/{filename}",
        prediction=prediction
    )

# ============================
# Run Flask
# ============================

if __name__ == "__main__":
    app.run(debug=True)