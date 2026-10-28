"""
ResNet-50 Transfer Learning Model Architecture
Implements:
- Input Image: (224, 224, 3)
- Backbone: Pretrained ResNet-50 (ImageNet weights, include_top=False)
- GlobalAveragePooling2D
- Dense(256, ReLU)
- Dropout(0.5)
- Dense(num_classes, Softmax)
- Dynamic class dimensioning
- Stage 1 (Feature Extraction) & Stage 2 (Fine-Tuning) configuration
"""

import json
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def build_resnet50_model(num_classes, learning_rate=1e-4, input_shape=(224, 224, 3)):
    """
    Constructs the ResNet-50 transfer learning architecture.
    Stage 1: Backbone frozen for feature extraction.
    """
    print(f"Building ResNet-50 architecture dynamically for {num_classes} classes...")

    # Load pretrained ResNet-50 backbone
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the backbone for initial feature extraction
    base_model.trainable = False

    # Define architecture graph
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling2d")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256_relu")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_0.5")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_MultiCrop_Disease_Classifier")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model, base_model


def configure_fine_tuning(model, base_model, fine_tune_at_layer=-20, fine_tune_lr=1e-5):
    """
    Stage 2 Fine-Tuning:
    Unfreezes deeper convolutional layers while keeping BatchNormalization in inference mode
    to prevent degradation of running statistics. Recompiles with lower learning rate.
    """
    print(f"Configuring Stage 2: Fine-Tuning from layer index {fine_tune_at_layer}...")
    base_model.trainable = True

    # Freeze all layers up to fine_tune_at_layer
    total_base_layers = len(base_model.layers)
    cutoff = total_base_layers + fine_tune_at_layer if fine_tune_at_layer < 0 else fine_tune_at_layer

    for i, layer in enumerate(base_model.layers[:cutoff]):
        layer.trainable = False

    # Keep BatchNormalization frozen during fine-tuning (best practice in transfer learning)
    for layer in base_model.layers[cutoff:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for layer in model.layers if layer.trainable)
    trainable_weights = len(model.trainable_weights)

    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    print(f"Fine-tuning enabled: {trainable_weights} trainable weight tensors ready.")
    return model


def save_model_summary(model, output_dir=RESULTS_DIR):
    """Saves text and structured JSON description of model architecture."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "model_architecture_summary.txt"
    with open(summary_file, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    def _get_shape(l):
        try:
            return str(getattr(l, "output_shape", getattr(l, "shape", "N/A")))
        except Exception:
            return "N/A"

    in_shape = "(None, 224, 224, 3)"
    out_shape = f"(None, {getattr(model.layers[-1], 'units', 'dynamic')})"

    architecture_info = {
        "model_name": model.name,
        "input_shape": in_shape,
        "output_shape": out_shape,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
        "non_trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)),
        "layers": [{"name": layer.name, "type": layer.__class__.__name__, "output_shape": _get_shape(layer)} for layer in model.layers]
    }

    with open(output_dir / "model_architecture.json", "w") as f:
        json.dump(architecture_info, f, indent=2)

    print(f"Saved model architecture summary to: {summary_file}")
    return architecture_info
