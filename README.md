# CIFAR-10 Image Classification with VGG16 Transfer Learning
**GTU 5th Semester • Data Science Project #18**

---

## 📌 Project Overview
This project classifies photographic images into 10 categories from the **CIFAR-10** dataset using **VGG16 Transfer Learning**. It features:
* Frozen pre-trained **ImageNet** convolutional backbone (14.7M parameters)
* Custom dense classification head with Batch Normalization and Dropout
* Interactive, modern **Web UI** for real-time image testing and evaluation
* ~80% validation accuracy achieved across 20 epochs

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Web UI (Recommended for Demo & Viva)
To start the web application and test predictions with images:

```powershell
python app.py
```

* Open your browser and go to: **`http://127.0.0.1:5000`**
* **Features:**
  * Drag-and-drop or upload custom images to classify in real-time (~99 ms latency).
  * 10 quick-test CIFAR-10 sample image buttons for instant demonstrations.
  * Live confidence scores and full probability breakdown for all 10 classes.
  * Full-screen gallery of all evaluation charts and architecture specifications.

---

### 2. Run the Full Model Training Pipeline (From Scratch or Resume)
If you want to train or evaluate the model from the command line:

```powershell
python cifar10_classification.py
```

* The script automatically uses saved checkpoints in `checkpoints/`.
* Generates 4 visualization plots in the project directory:
  * `training_history.png`
  * `confusion_matrix.png`
  * `feature_maps.png`
  * `sample_predictions.png`

---

## 📦 Requirements & Installation
If running on a new computer or virtual environment:

```powershell
pip install -r requirements.txt
pip install flask pillow
```

---

## 📁 Project Directory Structure
```text
DS-Project/
├── app.py                      # Flask Web UI Server
├── cifar10_classification.py   # Core Training & Evaluation Pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── checkpoints/                # Saved trained model & weights
│   ├── best_model.weights.h5   # Best epoch weights (~79.5% accuracy)
│   ├── model_weights.weights.h5# Final epoch weights
│   ├── checkpoint_meta.json    # Training summary metadata
│   ├── predictions.npz         # Saved evaluation predictions
│   └── training_history.pkl    # Full 20-epoch history
│
├── templates/
│   └── index.html              # Modern Web Dashboard
│
├── static/
│   ├── style.css               # Glassmorphic dark theme CSS
│   ├── app.js                  # Client-side JavaScript
│   └── samples/                # 10 test CIFAR-10 sample images
│
└── *.png                       # Generated evaluation plots
    ├── training_history.png
    ├── confusion_matrix.png
    ├── feature_maps.png
    └── sample_predictions.png
```
