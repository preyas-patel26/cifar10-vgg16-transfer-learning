"""
Flask Web Application for CIFAR-10 Image Classification
GTU 5th Sem Data Science Project
======================================================
Serves an interactive modern web UI for image classification
using the pre-trained VGG16 transfer learning model.
DOES NOT MODIFY cifar10_classification.py!
"""

import os
import io
import time
import json
import numpy as np
from PIL import Image

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'samples')
INPUT_SIZE = 64
NUM_CLASSES = 10

CLASS_NAMES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

CLASS_ICONS = {
    'Airplane': '✈️',
    'Automobile': '🚗',
    'Bird': '🐦',
    'Cat': '🐱',
    'Deer': '🦌',
    'Dog': '🐶',
    'Frog': '🐸',
    'Horse': '🐴',
    'Ship': '🚢',
    'Truck': '🚚'
}

# Global model instance
model = None


def load_classification_model():
    """Build exact VGG16 model architecture and load saved checkpoint weights."""
    global model
    print("[Web UI] Initializing VGG16 Transfer Learning Model...")

    base_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3)
    )
    base_model.trainable = False

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(32, 32, 3), name="cifar_input")
    x = layers.Resizing(INPUT_SIZE, INPUT_SIZE, name="resize")(inputs)
    x = data_augmentation(x)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation='relu', name="dense_256")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.5, name="dropout_50")(x)
    x = layers.Dense(128, activation='relu', name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_30")(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)

    m = keras.Model(inputs, outputs, name="CIFAR10_VGG16")

    # Determine best available weights
    weights_path = os.path.join(CHECKPOINT_DIR, 'best_model.weights.h5')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(CHECKPOINT_DIR, 'model_weights.weights.h5')

    if os.path.exists(weights_path):
        m.load_weights(weights_path)
        print(f"[Web UI] Successfully loaded trained weights from: {weights_path}")
    else:
        print("[Web UI] WARNING: No checkpoint weights found in checkpoints/")

    # Warm up model to eliminate cold-start inference lag
    dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
    _ = m.predict(dummy, verbose=0)
    print("[Web UI] Model warm-up complete. Ready for inference!")
    return m


def preprocess_pil_image(pil_img):
    """Preprocess PIL image into (1, 32, 32, 3) float32 numpy array in [0, 255] range."""
    # Convert to RGB (handles RGBA or Grayscale)
    rgb_img = pil_img.convert('RGB')
    resized_img = rgb_img.resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.array(resized_img, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    return batch


@app.route('/')
def index():
    """Serve the main Web UI dashboard."""
    # Read metadata if present
    meta_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    return render_template('index.html', meta=meta, class_names=CLASS_NAMES, class_icons=CLASS_ICONS)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict class of uploaded image."""
    global model
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read()))
        batch = preprocess_pil_image(pil_img)

        start_time = time.time()
        preds = model.predict(batch, verbose=0)[0]
        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        top_prob = float(preds[top_idx])
        top_icon = CLASS_ICONS.get(top_class, '🖼️')

        # Format full breakdown sorted descending
        breakdown = []
        for i in range(NUM_CLASSES):
            prob = float(preds[i])
            breakdown.append({
                'class': CLASS_NAMES[i],
                'icon': CLASS_ICONS.get(CLASS_NAMES[i], '🖼️'),
                'probability': prob,
                'percentage': f"{prob * 100:.1f}%",
                'is_top': (i == top_idx)
            })
        breakdown.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'prediction': top_class,
            'icon': top_icon,
            'confidence': round(top_prob * 100, 2),
            'breakdown': breakdown,
            'inference_time_ms': inference_time_ms
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict-sample/<name>')
def predict_sample(name):
    """Predict class for a pre-saved sample image."""
    global model
    sample_file = f"{name.lower()}.png"
    sample_path = os.path.join(SAMPLES_DIR, sample_file)

    if not os.path.exists(sample_path):
        return jsonify({'success': False, 'error': f'Sample {sample_file} not found'}), 404

    try:
        pil_img = Image.open(sample_path)
        batch = preprocess_pil_image(pil_img)

        start_time = time.time()
        preds = model.predict(batch, verbose=0)[0]
        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        top_prob = float(preds[top_idx])
        top_icon = CLASS_ICONS.get(top_class, '🖼️')

        breakdown = []
        for i in range(NUM_CLASSES):
            prob = float(preds[i])
            breakdown.append({
                'class': CLASS_NAMES[i],
                'icon': CLASS_ICONS.get(CLASS_NAMES[i], '🖼️'),
                'probability': prob,
                'percentage': f"{prob * 100:.1f}%",
                'is_top': (i == top_idx)
            })
        breakdown.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'sample_name': name,
            'sample_url': f'/static/samples/{sample_file}',
            'prediction': top_class,
            'icon': top_icon,
            'confidence': round(top_prob * 100, 2),
            'breakdown': breakdown,
            'inference_time_ms': inference_time_ms
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/results/<path:filename>')
def serve_result_image(filename):
    """Serve the generated visualization plots."""
    return send_from_directory(BASE_DIR, filename)


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    model = load_classification_model()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n==================================================")
    print(f">> CIFAR-10 Web UI is running at http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
