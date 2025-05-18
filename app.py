import os
import gdown
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Path where model will be stored after download
MODEL_PATH = "model.h5"
GOOGLE_DRIVE_ID = "1UqcY2fnDeeYvN52-vYhdQkOsvndZN0eE"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model with gdown...")
        gdown.download(id=GOOGLE_DRIVE_ID, output=MODEL_PATH, quiet=False)
        print("Model download complete!")

download_model()

# Load the model normally (no compile=False needed with updated TF)
model = load_model(MODEL_PATH)

# Define target size (adjust based on your model's input shape)
TARGET_SIZE = (224, 224)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file temporarily
    file_path = "temp.jpg"
    file.save(file_path)

    # Load and preprocess image
    img = image.load_img(file_path, target_size=TARGET_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize if your model expects it

    # Make prediction
    preds = model.predict(x)
    prediction = preds.argmax(axis=-1)[0]  # Adjust depending on model output

    # Clean up
    os.remove(file_path)

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
