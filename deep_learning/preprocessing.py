"""
Image Preprocessing and Efficient Data Loading Pipeline for ResNet-50
- 224 x 224 resolution
- RGB standardization
- Official ResNet-50 preprocessing (tf.keras.applications.resnet50.preprocess_input)
- Train-only data augmentation (rotation, zoom, flip, brightness)
- tf.data batching and prefetching with AUTOTUNE
"""

import os
import tensorflow as tf
from pathlib import Path

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation_layer():
    """
    Builds data augmentation pipeline applied exclusively during training.
    Maintains biological disease feature integrity without extreme distortions.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),  # approx +/- 15 degrees
        tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.05)),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ], name="data_augmentation")


def load_and_preprocess_image(file_path, label, is_training=False, num_classes=None, augmentation_layer=None):
    """
    Reads an image file from disk, decodes, resizes, optionally augments,
    and applies official ResNet-50 preprocessing.
    """
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    if is_training and augmentation_layer is not None:
        img = tf.expand_dims(img, 0)
        img = augmentation_layer(img, training=True)
        img = tf.squeeze(img, 0)

    # Official ResNet-50 preprocessing (zero-centers with ImageNet mean BGR)
    img = tf.keras.applications.resnet50.preprocess_input(img)

    if num_classes is not None:
        one_hot_label = tf.one_hot(label, depth=num_classes)
        return img, one_hot_label
    return img, label


def create_dataset_from_directory(data_dir, batch_size=BATCH_SIZE, is_training=False, class_names=None):
    """
    Builds high-performance tf.data pipeline with prefetching and batching.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    if class_names is None:
        class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    file_paths = []
    labels = []

    for cname in class_names:
        cdir = data_dir / cname
        if not cdir.exists():
            continue
        c_files = [str(p) for p in cdir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        file_paths.extend(c_files)
        labels.extend([class_to_idx[cname]] * len(c_files))

    if not file_paths:
        raise ValueError(f"No valid images found in {data_dir}.")

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(file_paths), 2000), seed=42)

    aug_layer = build_augmentation_layer() if is_training else None

    ds = ds.map(
        lambda x, y: load_and_preprocess_image(
            x, y, is_training=is_training, num_classes=num_classes, augmentation_layer=aug_layer
        ),
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds, class_names, len(file_paths)


def preprocess_single_image(image_input):
    """
    Preprocesses a single image (PIL Image, numpy array, or file path)
    for model inference. Returns array with batch dimension (1, 224, 224, 3).
    """
    from PIL import Image
    import numpy as np

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    else:
        img = image_input

    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Official ResNet-50 preprocessing
    preprocessed = tf.keras.applications.resnet50.preprocess_input(img_batch)
    return preprocessed
