import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

app = Flask(__name__)


try:
    print("🔄 Loading model...")
    loaded_model = tf.keras.models.load_model(
        "cat_dog_classifier_model.keras",
        custom_objects={'KerasLayer': hub.KerasLayer},
        safe_mode=False
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    loaded_model = None


@app.route('/')
def home():
    return render_template('index.html')  # ✅ serve your HTML file


@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        img_resized = cv2.resize(img, (224, 224))
        img_scaled = img_resized.astype('float32') / 255.0
        img_reshaped = np.expand_dims(img_scaled, axis=0)

        prediction = loaded_model.predict(img_reshaped)
        confidence = float(prediction[0][0])
        predicted_class = 'Cat' if confidence > 0.5 else 'Dog'

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
