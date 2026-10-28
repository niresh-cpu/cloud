"""
Inference Pipeline for ResNet-50 Multi-Crop Disease Classifier
Features:
- Singleton model caching
- Full leaf validation and file integrity checks
- ResNet-50 inference with latency benchmarking
- Uncertainty quantification for out-of-distribution detection
- Top-3 predictions and agricultural advisory integration
"""

import time
import json
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from deep_learning.preprocessing import preprocess_single_image
from deep_learning.leaf_validator import validate_image_file, is_valid_plant_leaf, assess_prediction_uncertainty

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_FILE = MODELS_DIR / "crop_disease_resnet50.keras"
TFLITE_FILE = MODELS_DIR / "crop_disease_resnet50.tflite"
CLASS_FILE = MODELS_DIR / "class_names.json"
INFO_FILE = DATA_DIR / "disease_information.json"

_cached_model = None
_cached_classes = None
_cached_disease_info = None
_cached_interpreter = None
_cached_in_idx = None
_cached_out_idx = None


def get_tflite_engine():
    """Initializes or returns cached ultra-fast TFLite interpreter (< 30ms latency)."""
    global _cached_interpreter, _cached_in_idx, _cached_out_idx
    if _cached_interpreter is None and TFLITE_FILE.exists():
        try:
            interp = tf.lite.Interpreter(model_path=str(TFLITE_FILE), num_threads=4)
            interp.allocate_tensors()
            _cached_interpreter = interp
            _cached_in_idx = interp.get_input_details()[0]["index"]
            _cached_out_idx = interp.get_output_details()[0]["index"]
            print("TFLite XNNPACK engine loaded successfully.")
        except Exception as e:
            print(f"Notice: TFLite engine could not load, using Keras: {e}")
            _cached_interpreter = None
    return _cached_interpreter, _cached_in_idx, _cached_out_idx


def load_inference_assets():
    """Loads and caches model, class names, and disease information database."""
    global _cached_model, _cached_classes, _cached_disease_info

    # Pre-warm TFLite engine if present
    get_tflite_engine()

    if _cached_model is None and _cached_interpreter is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_FILE}. Please run training first.")
        print(f"Loading ResNet-50 model from: {MODEL_FILE}")
        _cached_model = tf.keras.models.load_model(MODEL_FILE)

    if _cached_classes is None:
        if not CLASS_FILE.exists():
            raise FileNotFoundError(f"Class names mapping not found at {CLASS_FILE}.")
        with open(CLASS_FILE, "r") as f:
            _cached_classes = json.load(f)

    if _cached_disease_info is None:
        if INFO_FILE.exists():
            with open(INFO_FILE, "r") as f:
                _cached_disease_info = json.load(f)
        else:
            _cached_disease_info = {}

    return _cached_model, _cached_classes, _cached_disease_info


def predict_image(image_input, crop_filter=None):
    """
    Complete prediction workflow:
    1. Validate image format & dimensions
    2. Check leaf / plant validity (< 2ms)
    3. Run ResNet-50 inference via high-speed TFLite XNNPACK (< 30ms) or Keras callable
    4. Optional Crop Conditioning: if crop_filter is specified (e.g. 'Tomato'),
       normalizes predictions specifically over candidate diseases of that crop.
    5. Calculate uncertainty, top-3 candidates, and crop distribution breakdown
    6. Return enriched diagnostic result with instant latency
    """
    start_time = time.time()

    # 1. File & Format Validation
    if isinstance(image_input, (bytes, bytearray)):
        valid_file, msg, pil_img = validate_image_file(image_input)
    elif isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as f:
            valid_file, msg, pil_img = validate_image_file(f.read())
    elif isinstance(image_input, Image.Image):
        valid_file, msg, pil_img = True, "Valid image", image_input.convert("RGB")
    else:
        return {
            "success": False,
            "status": "invalid_input",
            "message": "Unsupported image input type.",
            "inference_time_ms": 0.0
        }

    if not valid_file:
        return {
            "success": False,
            "status": "invalid_file",
            "message": msg,
            "inference_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    # 2. Plant / Leaf Validation
    is_plant, plant_msg, _ = is_valid_plant_leaf(pil_img)
    if not is_plant:
        return {
            "success": False,
            "status": "not_a_plant",
            "message": "Invalid Image: Please upload a valid plant or leaf image.",
            "details": plant_msg,
            "inference_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    # 3. Model Inference via TFLite with Test-Time Augmentation (TTA) or fallback
    model, class_names, disease_info = load_inference_assets()
    preprocessed = preprocess_single_image(pil_img)

    interp, in_idx, out_idx = get_tflite_engine()
    if interp is not None:
        # View 1: Standard orientation
        interp.set_tensor(in_idx, preprocessed)
        interp.invoke()
        probs1 = interp.get_tensor(out_idx)[0].copy()

        # View 2: Horizontal mirror flip (TTA) for invariant feature extraction
        try:
            from PIL import ImageOps
            flipped_img = ImageOps.mirror(pil_img)
            prep_flipped = preprocess_single_image(flipped_img)
            interp.set_tensor(in_idx, prep_flipped)
            interp.invoke()
            probs2 = interp.get_tensor(out_idx)[0].copy()
            probs = (probs1 * 0.55 + probs2 * 0.45)
        except Exception:
            probs = probs1
    else:
        probs = model(preprocessed, training=False).numpy()[0].copy()

    # Calculate crop distribution breakdown across all 11 crops
    crop_scores = {}
    for idx, cname in enumerate(class_names):
        c_crop = cname.split("___")[0].replace("_", " ")
        crop_scores[c_crop] = crop_scores.get(c_crop, 0.0) + float(probs[idx])
    sorted_crops = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
    crop_breakdown = [{"crop": c, "score": round(s * 100, 1)} for c, s in sorted_crops if s > 0.01][:5]

    # 4. Optional Crop Conditioning (e.g. user specifies "Tomato" or "Pepper")
    active_crop_filter = None
    if crop_filter and str(crop_filter).strip().lower() not in {"auto", "all", "none", "", "[object object]"}:
        target = str(crop_filter).strip().lower()
        target_clean = target.replace(",", " ").replace("(", " ").replace(")", " ").replace("_", " ")
        target_words = [w for w in target_clean.split() if len(w) >= 3]

        indices_for_crop = []
        for i, c in enumerate(class_names):
            crop_name = c.split("___")[0].replace("_", " ")
            crop_name_lower = crop_name.lower()
            crop_clean = crop_name_lower.replace(",", " ").replace("(", " ").replace(")", " ")
            crop_words = [w for w in crop_clean.split() if len(w) >= 3]

            if (
                crop_name_lower == target
                or target in crop_name_lower
                or crop_name_lower in target
                or any(tw in crop_words for tw in target_words)
            ):
                indices_for_crop.append(i)

        if indices_for_crop:
            active_crop_filter = class_names[indices_for_crop[0]].split("___")[0].replace("_", " ")
            crop_sum = float(np.sum(probs[indices_for_crop]))
            if crop_sum > 0:
                conditioned_probs = np.zeros_like(probs)
                conditioned_probs[indices_for_crop] = probs[indices_for_crop] / crop_sum
                probs = conditioned_probs

    # 4b. Soft Bayesian calibration for cross-crop consistency:
    # Suppresses spurious cross-species noise while preserving intra-crop pathology hierarchy.
    if not active_crop_filter:
        crop_groups = {}
        for i, cname in enumerate(class_names):
            cr = cname.split("___")[0]
            if cr not in crop_groups:
                crop_groups[cr] = []
            crop_groups[cr].append(i)

        calibrated_probs = probs.copy()
        for cr, idxs in crop_groups.items():
            c_mass = float(np.sum(probs[idxs]))
            calibrated_probs[idxs] *= (c_mass ** 0.30)
        prob_sum = float(np.sum(calibrated_probs))
        if prob_sum > 0:
            probs = calibrated_probs / prob_sum

    # 5. Uncertainty Check
    is_confident, uncertainty_msg = assess_prediction_uncertainty(probs)
    if not is_confident and not active_crop_filter:
        top_idx = int(np.argmax(probs))
        top_class = class_names[top_idx]
        return {
            "success": False,
            "status": "uncertain_prediction",
            "message": uncertainty_msg,
            "provisional_class": top_class,
            "confidence": float(round(probs[top_idx], 4)),
            "crop_breakdown": crop_breakdown,
            "inference_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    # 6. Extract Top-3 Predictions
    top3_indices = np.argsort(probs)[::-1][:3]
    top_predictions = []
    for idx in top3_indices:
        if probs[idx] <= 0.0001 and len(top_predictions) > 0:
            continue
        cname = class_names[idx]
        parts = cname.split("___")
        crop = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        top_predictions.append({
            "class_id": cname,
            "crop": crop,
            "disease": disease,
            "confidence": float(round(probs[idx], 4)),
            "confidence_percentage": f"{round(float(probs[idx]) * 100, 2)}%"
        })

    # Best prediction
    primary = top_predictions[0]
    cname = primary["class_id"]
    parts = cname.split("___")
    crop_name = parts[0].replace("_", " ")
    disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease_name.lower()

    # Get structured advisory
    info = disease_info.get(cname, {})
    description = info.get("description", "Identified crop leaf disease.")
    symptoms = info.get("symptoms", ["Inspect leaf foliage for spot lesions or chlorosis."])
    prevention = info.get("prevention", ["Maintain proper plant spacing and drip irrigation."])
    management = info.get("management", ["Consult regional agricultural extension services."])
    pathogen_type = info.get("pathogen_type", "Unknown")
    severity = info.get("severity", "Medium" if not is_healthy else "None")

    inference_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "success": True,
        "status": "success",
        "crop": crop_name,
        "disease": disease_name,
        "class_id": cname,
        "is_healthy": is_healthy,
        "status_label": "Healthy" if is_healthy else "Diseased",
        "confidence": primary["confidence"],
        "confidence_percentage": primary["confidence_percentage"],
        "top_predictions": top_predictions,
        "pathogen_type": pathogen_type,
        "severity": severity,
        "description": description,
        "symptoms": symptoms,
        "prevention": prevention,
        "management": management,
        "crop_filter": active_crop_filter,
        "crop_breakdown": crop_breakdown,
        "inference_time_ms": inference_ms,
        "validation_status": "Valid plant leaf confirmed"
    }
