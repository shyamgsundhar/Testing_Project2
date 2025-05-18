from flask import Flask, request, jsonify
import os
import requests
import tensorflow as tf
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model.h5"
GOOGLE_DRIVE_ID = "1UqcY2fnDeeYvN52-vYhdQkOsvndZN0eE"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model download complete!")


download_model()

# Load model after download
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_array = np.array(data["input"]).reshape(1, -1)
    prediction = model.predict(input_array)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
