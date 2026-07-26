from gradcam import generate_gradcam, save_gradcam
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from database import (
    init_db,
    save_prediction,
    get_history,
    get_total_predictions,
    get_normal_cases,
    get_pneumonia_cases,
    get_latest_prediction
)
from gemini_helper import generate_medical_report
from datetime import datetime
import tensorflow as tf
import os
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

"""
AI Medical Intelligence Platform

Features:
1. Medicine Recommendation System
2.Chest X-ray Disease Detection

"""
app = Flask(__name__)
init_db()
# -----------------------------
# Upload Folder
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs("static/gradcam", exist_ok=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# Medicine Recommendation
# -----------------------------
try:
    vectorizer = joblib.load("model/vectorizer.pkl")
    vectors = joblib.load("model/vectors.pkl")
    df = joblib.load("model/medicine_data.pkl")
    xray_model = load_model("model/xray_model.keras")
except Exception as e:
    print("Error loading models:", e)
    raise

IMG_SIZE = (224, 224)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # =============================
    # Medicine Recommendation
    # =============================
    if "medicine" in request.form and request.form["medicine"].strip() != "":

        medicine = request.form["medicine"]

        vector = vectorizer.transform([medicine])
        similarity = cosine_similarity(vector, vectors)

        indices = similarity[0].argsort()[-6:][::-1]

        recommendations = []

        for i in indices:

            row = df.iloc[i]

            if row["name"].lower() != medicine.lower():

                recommendations.append({

                    "name": row["name"],

                    "uses":[
                        row["use0"],
                        row["use1"],
                        row["use2"],
                        row["use3"],
                        row["use4"]
                    ],

                    "chemical": row["Chemical Class"],

                    "therapeutic": row["Therapeutic Class"],

                    "action": row["Action Class"],

                    "habit": row["Habit Forming"],

                    "side_effects":[
                        row["sideEffect0"],
                        row["sideEffect1"],
                        row["sideEffect2"],
                        row["sideEffect3"],
                        row["sideEffect4"]
                    ]

                })

        recommendations = recommendations[:5]

        prediction_time = datetime.now().strftime("%d-%m-%Y %H:%M")


        return render_template(
            "result.html",
            medicine=medicine,
            recommendations=recommendations,
            time=prediction_time
        )
    # =============================
    # Chest X-ray Prediction
    # =============================
    if "image" in request.files:

        image = request.files["image"]

        if image.filename != "":

            filename = secure_filename(image.filename)

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(filepath)

            img = Image.open(filepath).convert("RGB")
            img = img.resize(IMG_SIZE)

            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            print("Model Inputs :", xray_model.inputs)
            print("Model Outputs:", xray_model.outputs)
            print("Last Conv Layer:", xray_model.get_layer("conv2d_2").output.shape)

            gradcam_filename = None

            prediction = xray_model.predict(img)[0][0]

       
            if prediction > 0.5:

                print("Prediction:", prediction)

                img = tf.convert_to_tensor(img, dtype=tf.float32)

                heatmap = generate_gradcam(xray_model, img)

                gradcam_filename = "gradcam_" + filename

                gradcam_path = os.path.join(
                    "static",
                    "gradcam",
                    gradcam_filename
                )

                save_gradcam(
                   filepath,
                   heatmap,
                   gradcam_path
                )

                disease = "PNEUMONIA"
                confidence = prediction * 100
            else:
                disease = "NORMAL"
                confidence = (1 - prediction) * 100
            print(filename, type(filename))
            print(disease, type(disease))
            print(confidence, type(confidence))

            save_prediction(
                str(filename),
                str(disease),
                float(round(confidence, 2))
            )

            print("Disease =", disease)
            print("Confidence =", confidence)

            prediction_time = datetime.now().strftime("%d-%m-%Y %H:%M")

            # Generate AI Medical Report using Gemini
            ai_report = generate_medical_report(
                   disease,
                   round(confidence, 2)
            )

            return render_template(
                "xray_result.html",
                disease=disease,
                confidence=round(confidence, 2),
                image="uploads/" + filename,
                gradcam=("gradcam/" + gradcam_filename) if gradcam_filename else None,
                time=prediction_time,
                model_accuracy=89.74,
                ai_report=ai_report
            )

        return render_template("index.html")


@app.route("/history")
def history():
    history = get_history()
    return render_template(
        "history.html",
        history=history
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model")
def model():
    return render_template("model_info.html")

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        total=get_total_predictions(),
        normal=get_normal_cases(),
        pneumonia=get_pneumonia_cases(),
        latest=get_latest_prediction()
    )

@app.route("/api/history")
def api_history():

    history = get_history()

    data = []

    for row in history:
        data.append({
            "id": row[0],
            "image": row[1],
            "disease": row[2],
            "confidence": row[3],
            "time": row[4]
        })

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)