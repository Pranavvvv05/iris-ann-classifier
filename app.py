import json

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")
with open("classes.json") as f:
    CLASSES = json.load(f)

DISPLAY_NAMES = {
    "Iris-setosa": "Setosa",
    "Iris-versicolor": "Versicolor",
    "Iris-virginica": "Virginica",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"]),
        ]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Send sepal_length, sepal_width, petal_length, petal_width as numbers."}), 400

    x = scaler.transform(np.array([features]))
    probs = model.predict(x, verbose=0)[0]

    result = [
        {"species": cls, "label": DISPLAY_NAMES.get(cls, cls), "confidence": round(float(p), 4)}
        for cls, p in zip(CLASSES, probs)
    ]
    result.sort(key=lambda r: r["confidence"], reverse=True)

    return jsonify({"prediction": result[0], "all": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
