"""
Two-Stage Training Pipeline for Multi-Crop Plant Leaf Disease Detection
Stage 1: Feature Extraction (ResNet-50 backbone frozen)
Stage 2: Fine-Tuning (Top convolutional blocks unfrozen with reduced learning rate)
Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
Generates:
- models/crop_disease_resnet50.keras
- models/class_names.json
- results/training_history.csv
- results/accuracy_curve.png
- results/loss_curve.png
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from deep_learning.preprocessing import create_dataset_from_directory, BATCH_SIZE
from deep_learning.model import build_resnet50_model, configure_fine_tuning, save_model_summary

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
TRAIN_DIR = DATASETS_DIR / "train"
VAL_DIR = DATASETS_DIR / "validation"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def plot_training_curves(history_combined, output_dir=RESULTS_DIR):
    """Generates publication-quality training & validation accuracy and loss graphs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_combined["accuracy"]) + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history_combined["accuracy"], "b-o", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, history_combined["val_accuracy"], "g-s", linewidth=2, label="Validation Accuracy")
    plt.title("ResNet-50 Multi-Crop Disease Classification Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    acc_path = output_dir / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Accuracy curve saved to: {acc_path}")

    # 2. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history_combined["loss"], "r-o", linewidth=2, label="Training Loss")
    plt.plot(epochs, history_combined["val_loss"], "m-s", linewidth=2, label="Validation Loss")
    plt.title("ResNet-50 Multi-Crop Categorical Crossentropy Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to: {loss_path}")


def run_training(
    stage1_epochs=10,
    stage2_epochs=6,
    batch_size=32,
    stage1_lr=1e-4,
    stage2_lr=1e-5,
    fine_tune_layers=-20
):
    print("=================================================================")
    print("STARTING RESNET-50 TWO-STAGE MULTI-CROP DISEASE MODEL TRAINING")
    print("=================================================================")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_ds, class_names, num_train_imgs = create_dataset_from_directory(
        TRAIN_DIR, batch_size=batch_size, is_training=True
    )
    val_ds, _, num_val_imgs = create_dataset_from_directory(
        VAL_DIR, batch_size=batch_size, is_training=False, class_names=class_names
    )

    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes from training dataset.")
    print(f"Train samples: {num_train_imgs} | Validation samples: {num_val_imgs}")

    # Save synchronized class_names.json
    class_mapping_path = MODELS_DIR / "class_names.json"
    with open(class_mapping_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Synchronized class names saved to: {class_mapping_path}")

    # 2. Build Stage 1 Model
    model, base_model = build_resnet50_model(num_classes=num_classes, learning_rate=stage1_lr)
    save_model_summary(model, output_dir=RESULTS_DIR)

    best_model_path = MODELS_DIR / "crop_disease_resnet50.keras"

    # Callbacks
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
        )
    ]

    print("\n--- STAGE 1: Feature Extraction (Backbone Frozen) ---")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1,
        verbose=1
    )

    history_stage1 = h1.history

    # 3. Stage 2 Fine-Tuning
    history_stage2 = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    if stage2_epochs > 0:
        print("\n--- STAGE 2: Fine-Tuning Selected Deeper Layers ---")
        model = configure_fine_tuning(
            model, base_model, fine_tune_at_layer=fine_tune_layers, fine_tune_lr=stage2_lr
        )

        callbacks_stage2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path), monitor="val_accuracy", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
            )
        ]

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage2_epochs,
            callbacks=callbacks_stage2,
            verbose=1
        )
        history_stage2 = h2.history

    # 4. Merge Training History
    history_combined = {}
    for k in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        val1 = history_stage1.get(k, [])
        val2 = history_stage2.get(k, [])
        history_combined[k] = val1 + val2

    # Save training history CSV
    history_df = pd.DataFrame(history_combined)
    history_df.index = history_df.index + 1
    history_df.index.name = "epoch"
    history_csv = RESULTS_DIR / "training_history.csv"
    history_df.to_csv(history_csv)
    print(f"Saved full training history log to: {history_csv}")

    # Plot graphs
    plot_training_curves(history_combined, output_dir=RESULTS_DIR)

    # Save training configuration metadata
    best_val_acc = max(history_combined["val_accuracy"])
    best_val_epoch = history_combined["val_accuracy"].index(best_val_acc) + 1

    training_metadata = {
        "architecture": "ResNet-50 Transfer Learning",
        "input_shape": [224, 224, 3],
        "total_classes": num_classes,
        "batch_size": batch_size,
        "stage1_epochs_run": len(history_stage1.get("accuracy", [])),
        "stage2_epochs_run": len(history_stage2.get("accuracy", [])),
        "stage1_lr": stage1_lr,
        "stage2_lr": stage2_lr,
        "best_validation_accuracy": float(round(best_val_acc, 4)),
        "best_validation_epoch": int(best_val_epoch),
        "final_training_accuracy": float(round(history_combined["accuracy"][-1], 4)),
        "final_training_loss": float(round(history_combined["loss"][-1], 4)),
        "final_validation_loss": float(round(history_combined["val_loss"][-1], 4))
    }

    with open(RESULTS_DIR / "training_config.json", "w") as f:
        json.dump(training_metadata, f, indent=2)

    print("\n=================================================================")
    print(f"TRAINING COMPLETE. Best Val Accuracy: {round(best_val_acc * 100, 2)}% at epoch {best_val_epoch}")
    print(f"Saved Model: {best_model_path}")
    print("=================================================================")

    return training_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 Multi-Crop Disease Classifier")
    parser.add_argument("--stage1_epochs", type=int, default=8, help="Stage 1 feature extraction epochs")
    parser.add_argument("--stage2_epochs", type=int, default=4, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_training(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size
    )
