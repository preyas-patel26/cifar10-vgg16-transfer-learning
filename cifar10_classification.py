"""
Project #18: Image Classification with CIFAR-10
=================================================
Objective: Classify multi-class photographic categories using VGG16 transfer learning.

Implementation:
  - VGG16 backbone with frozen ImageNet weights
  - Custom dense classification head with categorical cross-entropy
  - Feature map visualization of internal layer activations

Dataset: CIFAR-10 (auto-downloaded by Keras)
"""

import os

# Suppress TF info/warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI popups)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
INPUT_SIZE = 64            # Balanced: large enough for VGG16, fast enough on CPU
NUM_CLASSES = 10
VALIDATION_SPLIT = 5000

CLASS_NAMES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')


# ============================================================
# Checkpoint Save / Load Helpers
# ============================================================
def save_checkpoint(model, history, y_pred, y_pred_probs, y_true, step_completed):
    """Save everything needed to resume from any step."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save model weights
    model.save_weights(os.path.join(CHECKPOINT_DIR, 'model_weights.weights.h5'))
    print(f"  [Checkpoint] Model weights saved.")

    # Save training history
    if history is not None:
        with open(os.path.join(CHECKPOINT_DIR, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
        print(f"  [Checkpoint] Training history saved.")

    # Save predictions
    if y_pred is not None:
        np.savez(os.path.join(CHECKPOINT_DIR, 'predictions.npz'),
                 y_pred=y_pred, y_pred_probs=y_pred_probs, y_true=y_true)
        print(f"  [Checkpoint] Predictions saved.")

    # Save progress metadata
    meta = {
        'step_completed': step_completed,
        'epochs_trained': len(history.history['loss']) if history else 0,
        'final_train_acc': float(history.history['accuracy'][-1]) if history else 0,
        'final_val_acc': float(history.history['val_accuracy'][-1]) if history else 0,
    }
    with open(os.path.join(CHECKPOINT_DIR, 'checkpoint_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  [Checkpoint] Progress saved at step {step_completed}.")
    print()


def load_checkpoint():
    """Load checkpoint metadata. Returns None if no checkpoint exists."""
    meta_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_meta.json')
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    print(f"  [Checkpoint] Found saved checkpoint!")
    print(f"    Last completed step : {meta['step_completed']}")
    print(f"    Epochs trained      : {meta['epochs_trained']}")
    print(f"    Train accuracy      : {meta['final_train_acc']:.4f}")
    print(f"    Val accuracy        : {meta['final_val_acc']:.4f}")
    print()
    return meta


def load_saved_history():
    """Load training history from checkpoint."""
    path = os.path.join(CHECKPOINT_DIR, 'training_history.pkl')
    with open(path, 'rb') as f:
        hist_dict = pickle.load(f)
    # Wrap in a simple object so history.history works like Keras history
    class HistoryObj:
        def __init__(self, d):
            self.history = d
    return HistoryObj(hist_dict)


def load_saved_predictions():
    """Load saved predictions from checkpoint."""
    data = np.load(os.path.join(CHECKPOINT_DIR, 'predictions.npz'))
    return data['y_pred'], data['y_pred_probs'], data['y_true']


# ============================================================
# Step 1: Load and Preprocess CIFAR-10
# ============================================================
def load_and_preprocess_data():
    """Load CIFAR-10 dataset, normalize, one-hot encode, and split."""
    print("=" * 60)
    print("STEP 1: Loading CIFAR-10 Dataset")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    print(f"  Training images  : {x_train.shape}")
    print(f"  Test images      : {x_test.shape}")
    print(f"  Pixel range      : [{x_train.min()}, {x_train.max()}]")

    # Keep pixel values as float32 in [0, 255] range
    # VGG16's preprocess_input expects [0, 255] and handles normalization itself
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(f"  One-hot example  : label {y_train[0][0]} -> {y_train_cat[0]}")

    # Split training into train + validation
    x_val = x_train[-VALIDATION_SPLIT:]
    y_val_cat = y_train_cat[-VALIDATION_SPLIT:]
    x_train_split = x_train[:len(x_train) - VALIDATION_SPLIT]
    y_train_split = y_train_cat[:len(y_train_cat) - VALIDATION_SPLIT]

    print(f"  Train / Val / Test: {len(x_train_split)} / {len(x_val)} / {len(x_test)}")
    print()

    return x_train_split, y_train_split, x_val, y_val_cat, x_test, y_test_cat, y_test


# ============================================================
# Step 2: Data Augmentation Layer
# ============================================================
def create_augmentation_layer():
    """Create data augmentation pipeline (applied only during training)."""
    print("STEP 2: Creating Data Augmentation Pipeline")
    print("-" * 60)

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    print("  Augmentations: HorizontalFlip, Rotation(±18°), Zoom(±10%)")
    print()
    return data_augmentation


# ============================================================
# Step 3: Build Transfer Learning Model
# ============================================================
def build_model(data_augmentation):
    """Build VGG16 transfer learning model with custom classification head."""
    print("STEP 3: Building VGG16 Transfer Learning Model")
    print("-" * 60)

    # Load VGG16 backbone WITHOUT top classification layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3)
    )

    # Freeze all base model layers
    base_model.trainable = False
    print(f"  VGG16 layers     : {len(base_model.layers)} (ALL FROZEN)")
    print(f"  Base params      : {base_model.count_params():,} (non-trainable)")

    # Build the complete model pipeline
    inputs = keras.Input(shape=(32, 32, 3), name="cifar_input")

    # Resize 32×32 → 64×64
    x = layers.Resizing(INPUT_SIZE, INPUT_SIZE, name="resize")(inputs)

    # Data augmentation (active only during training)
    x = data_augmentation(x)

    # VGG16 preprocessing: RGB→BGR, zero-centers each channel
    # This expects input in [0, 255] range — which is what we pass
    x = tf.keras.applications.vgg16.preprocess_input(x)

    # Frozen VGG16 backbone
    x = base_model(x, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation='relu', name="dense_256")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.5, name="dropout_50")(x)
    x = layers.Dense(128, activation='relu', name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_30")(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)

    model = keras.Model(inputs, outputs, name="CIFAR10_VGG16")

    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {total - trainable:,}")
    print()

    model.summary()
    print()

    return model, base_model


# ============================================================
# Step 4: Compile Model
# ============================================================
def compile_model(model):
    """Compile with Adam optimizer and categorical cross-entropy loss."""
    print("STEP 4: Compiling Model")
    print("-" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"  Optimizer : Adam (lr={LEARNING_RATE})")
    print(f"  Loss      : Categorical Cross-Entropy")
    print(f"  Metrics   : Accuracy")
    print()


# ============================================================
# Step 5: Train Model
# ============================================================
def train_model(model, x_train, y_train, x_val, y_val):
    """Train the model with early stopping and learning rate reduction."""
    print("STEP 5: Training Model")
    print("-" * 60)
    print(f"  Max epochs   : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, 'best_model.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print()
    return history


# ============================================================
# Step 6: Plot Training History
# ============================================================
def plot_training_history(history):
    """Plot accuracy and loss curves."""
    print("STEP 6: Plotting Training History")
    print("-" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy',
             linewidth=2, color='#2196F3')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy',
             linewidth=2, color='#FF9800', linestyle='--')
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss',
             linewidth=2, color='#F44336')
    ax2.plot(history.history['val_loss'], label='Val Loss',
             linewidth=2, color='#4CAF50', linestyle='--')
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print()


# ============================================================
# Step 7: Evaluate on Test Set
# ============================================================
def evaluate_model(model, x_test, y_test_cat, y_test):
    """Evaluate and print classification report."""
    print("STEP 7: Evaluating on Test Set")
    print("-" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\n  Test Accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"  Test Loss     : {test_loss:.4f}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    print("\n  Classification Report:")
    print("  " + "-" * 58)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    for line in report.split('\n'):
        print(f"  {line}")
    print()

    return y_pred, y_pred_probs, y_true


# ============================================================
# Step 8: Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix heatmap."""
    print("STEP 8: Plotting Confusion Matrix")
    print("-" * 60)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='gray')
    plt.title('Confusion Matrix — CIFAR-10', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print()


# ============================================================
# Step 9: Feature Map Visualization (KEY REQUIREMENT)
# ============================================================
def visualize_feature_maps(base_model, x_test, y_test):
    """
    Extract and visualize internal layer feature maps from VGG16.
    Shows what convolutional filters detect at different depths:
      - Early layers: edges, colors, gradients
      - Middle layers: textures, shapes, patterns
      - Deep layers: high-level object parts
    """
    print("STEP 9: Visualizing Internal Layer Feature Maps")
    print("-" * 60)

    # Select layers from early, middle, and deep VGG16 blocks
    layer_names = ['block1_conv2', 'block3_conv3', 'block5_conv3']
    layer_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create sub-model for intermediate activations
    activation_model = keras.Model(
        inputs=base_model.input,
        outputs=layer_outputs
    )

    # Prepare test image — keep in [0, 255] for VGG16 preprocessing
    test_img = x_test[0]  # Shape: (32, 32, 3), range [0, 255]
    true_label = CLASS_NAMES[y_test[0][0]]

    # Resize to VGG16 input size
    test_img_resized = tf.image.resize(test_img, (INPUT_SIZE, INPUT_SIZE))

    # Apply VGG16 preprocessing (expects [0, 255])
    test_img_preprocessed = tf.keras.applications.vgg16.preprocess_input(
        test_img_resized[np.newaxis, ...]
    )

    # Get activations
    activations = activation_model.predict(test_img_preprocessed, verbose=0)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Original image (convert to [0, 1] for display)
    axes[0].imshow((test_img / 255.0).clip(0, 1))
    axes[0].set_title(f'Original Image\nClass: {true_label}', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Feature maps from each layer
    for idx, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        n_filters = min(16, activation.shape[-1])
        size = activation.shape[1]
        grid_size = 4

        display_grid = np.zeros((size * grid_size, size * grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                filter_idx = i * grid_size + j
                if filter_idx < n_filters:
                    channel_image = activation[0, :, :, filter_idx].copy()
                    if channel_image.std() > 0:
                        channel_image = (channel_image - channel_image.mean()) / channel_image.std()
                        channel_image = np.clip(channel_image * 64 + 128, 0, 255).astype('uint8')
                    display_grid[i * size:(i + 1) * size,
                                 j * size:(j + 1) * size] = channel_image

        axes[idx + 1].imshow(display_grid, cmap='viridis', aspect='auto')
        axes[idx + 1].set_title(
            f'{layer_name}\n{activation.shape[1]}×{activation.shape[2]}×{activation.shape[3]}',
            fontsize=11, fontweight='bold'
        )
        axes[idx + 1].axis('off')

    plt.suptitle('Feature Map Visualization — VGG16 Layer Activations',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'feature_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print("  Interpretation:")
    print("    block1_conv2 (Early) → Edges, color gradients, simple patterns")
    print("    block3_conv3 (Mid)   → Textures, corners, shapes")
    print("    block5_conv3 (Deep)  → High-level object parts, semantic features")
    print()


# ============================================================
# Step 10: Sample Predictions Display
# ============================================================
def plot_sample_predictions(x_test, y_true, y_pred, y_pred_probs):
    """Display sample test images with predictions."""
    print("STEP 10: Displaying Sample Predictions")
    print("-" * 60)

    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    np.random.seed(42)
    indices = np.random.choice(len(x_test), 15, replace=False)

    for ax, idx in zip(axes.flat, indices):
        ax.imshow((x_test[idx] / 255.0).clip(0, 1))
        true_label = CLASS_NAMES[y_true[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        confidence = y_pred_probs[idx][y_pred[idx]] * 100

        is_correct = true_label == pred_label
        color = '#2E7D32' if is_correct else '#C62828'
        symbol = 'Correct' if is_correct else 'Wrong'

        ax.set_title(
            f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%) [{symbol}]',
            color=color, fontsize=9, fontweight='bold'
        )
        ax.axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print()


# ============================================================
# Main Execution
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  CIFAR-10 Image Classification with VGG16 Transfer Learning")
    print("=" * 60 + "\n")

    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    last_step = checkpoint['step_completed'] if checkpoint else 0

    # --- Always needed: load data and build model ---
    # Step 1: Load data
    x_train, y_train, x_val, y_val, x_test, y_test_cat, y_test = \
        load_and_preprocess_data()

    # Step 2: Create augmentation
    data_augmentation = create_augmentation_layer()

    # Step 3: Build model
    model, base_model = build_model(data_augmentation)

    # Step 4: Compile
    compile_model(model)

    # --- Resume logic ---
    if last_step >= 5:
        # Training was already completed — restore weights and history
        print("  [Resume] Loading saved model weights...")
        model.load_weights(os.path.join(CHECKPOINT_DIR, 'model_weights.weights.h5'))
        history = load_saved_history()
        print(f"  [Resume] Restored {len(history.history['loss'])} epochs of training.")
        print()
    else:
        # Step 5: Train (fresh)
        history = train_model(model, x_train, y_train, x_val, y_val)
        # Save checkpoint after training
        save_checkpoint(model, history, None, None, None, step_completed=5)

    if last_step >= 6:
        print("  [Resume] Skipping Step 6 (training_history.png already exists).")
    else:
        # Step 6: Plot training curves
        plot_training_history(history)
        save_checkpoint(model, history, None, None, None, step_completed=6)

    if last_step >= 7:
        # Evaluation was done — load saved predictions
        print("  [Resume] Loading saved predictions...")
        y_pred, y_pred_probs, y_true = load_saved_predictions()
        print(f"  [Resume] Restored {len(y_pred)} predictions.")
        print()
    else:
        # Step 7: Evaluate
        y_pred, y_pred_probs, y_true = evaluate_model(model, x_test, y_test_cat, y_test)
        save_checkpoint(model, history, y_pred, y_pred_probs, y_true, step_completed=7)

    if last_step >= 8:
        print("  [Resume] Skipping Step 8 (confusion_matrix.png already exists).")
    else:
        # Step 8: Confusion matrix
        plot_confusion_matrix(y_true, y_pred)
        save_checkpoint(model, history, y_pred, y_pred_probs, y_true, step_completed=8)

    if last_step >= 9:
        print("  [Resume] Skipping Step 9 (feature_maps.png already exists).")
    else:
        # Step 9: Feature map visualization
        visualize_feature_maps(base_model, x_test, y_test)
        save_checkpoint(model, history, y_pred, y_pred_probs, y_true, step_completed=9)

    if last_step >= 10:
        print("  [Resume] Skipping Step 10 (sample_predictions.png already exists).")
    else:
        # Step 10: Sample predictions
        plot_sample_predictions(x_test, y_true, y_pred, y_pred_probs)
        save_checkpoint(model, history, y_pred, y_pred_probs, y_true, step_completed=10)

    # Done!
    print("=" * 60)
    print("  PROJECT COMPLETE!")
    print("=" * 60)
    print(f"\n  Output files in: {OUTPUT_DIR}")
    print("    - training_history.png")
    print("    - confusion_matrix.png")
    print("    - feature_maps.png")
    print("    - sample_predictions.png")
    print(f"    - checkpoints/  (for resume capability)")
    print()


if __name__ == "__main__":
    main()
