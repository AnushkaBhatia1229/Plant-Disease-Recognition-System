from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import os
import tensorflow as tf

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model(
    "plant_disease_recog_model_pwp.keras",
    compile=False
)

# Load disease information
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/uploads/<path:filename>")
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")
def home():
    return render_template("home.html")


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


if __name__ == "__main__":
    app.run(debug=True)