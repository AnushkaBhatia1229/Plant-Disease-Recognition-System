from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import os
import gc

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import gdown


app = Flask(__name__)


# =====================================
# TensorFlow Memory Optimization
# =====================================

try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass


# =====================================
# Upload Configuration
# =====================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024



# =====================================
# Model Download
# =====================================

MODEL_PATH = "plant_disease_model.keras"


if not os.path.exists(MODEL_PATH):

    print("Downloading AI Model...")

    gdown.download(
        id="14BUv71OaCFl4_b0ypL1rdob-zM_Cuwjd",
        output=MODEL_PATH,
        quiet=False
    )


print("Model file ready")



# =====================================
# Lazy Model Loading
# =====================================

model = None


def load_model_once():

    global model

    if model is None:

        print("Loading TensorFlow model...")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("Model Loaded Successfully")


    return model



# =====================================
# Disease Labels
# =====================================

with open(
    "plant_disease.json",
    "r"
) as file:

    plant_disease = json.load(file)



# =====================================
# Routes
# =====================================


@app.route("/uploads/<path:filename>")
def uploaded_images(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )



@app.route("/")
def home():

    return render_template(
        "home.html"
    )



# =====================================
# Image Processing
# =====================================


def extract_features(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(160,160)
    )


    image = tf.keras.utils.img_to_array(
        image
    )


    image = np.expand_dims(
        image,
        axis=0
    )


    return image




def model_predict(image_path):

    model = load_model_once()


    img = extract_features(
        image_path
    )


    prediction = model.predict(
        img,
        verbose=0
    )


    index = np.argmax(
        prediction
    )


    result = plant_disease[index]


    # Free memory
    del img
    del prediction

    gc.collect()


    return result




# =====================================
# Upload Prediction
# =====================================


@app.route(
    "/upload/",
    methods=["POST"]
)

def uploadimage():


    if "img" not in request.files:

        return redirect("/")



    image = request.files["img"]



    if image.filename == "":

        return redirect("/")



    filename = (
        f"temp_{uuid.uuid4().hex}_{image.filename}"
    )


    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    image.save(filepath)



    prediction = model_predict(
        filepath
    )



    return render_template(
        "home.html",
        result=True,
        imagepath=f"/uploads/{filename}",
        prediction=prediction
    )



# =====================================
# Run
# =====================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )