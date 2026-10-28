"""
Plant / Leaf Image Validator & Out-Of-Distribution Uncertainty Guard
Prevents non-plant images (cars, people, buildings, animals, video game graphics,
cinematic posters, fantasy art, charts, documents, objects) from producing false plant disease predictions.
Workflows:
1. File structure & header validation (multi-format support)
2. Darkness / cinematic scenery & shadow thresholding
3. Synthetic / document / chart / monochrome canvas detection
4. Semantic plant / flora verification via MobileNetV2 top-category filtering & whitelist
5. Multi-spectral foliage chromaticity analysis (chlorophyll / chlorosis / necrosis)
6. Out-Of-Distribution uncertainty quantification via prediction entropy & confidence margin
"""

import io
import math
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

BASE_DIR = Path(__file__).resolve().parent.parent
MOBILENET_TFLITE_FILE = BASE_DIR / "models" / "mobilenet_v2.tflite"

# Classes from ImageNet that can legitimately represent plant leaves, crops, produce,
# or organic macro leaf textures:
ALLOWED_LEAF_TOP1_CLASSES = {
    # Direct plants & produce
    'head_cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti_squash',
    'acorn_squash', 'butternut_squash', 'cucumber', 'artichoke', 'bell_pepper',
    'cardoon', 'mushroom', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple',
    'banana', 'jackfruit', 'custard_apple', 'pomegranate', 'hay', 'daisy', 'corn',
    'acorn', 'rapeseed', "yellow_lady's_slipper", 'hip', 'buckeye', 'agaric',
    'gyromitra', 'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear',
    'Granny_Smith', 'pot', 'vase', 'greenhouse',
    # Leaf insects / critters that dwell on foliage in macro photography
    'leaf_beetle', 'tree_frog', 'African_chameleon', 'American_chameleon',
    'grasshopper', 'leafhopper', 'cabbage_butterfly', 'sulphur_butterfly', 'lacewing',
    'ant', 'cricket', 'vine_snake', 'green_mamba', 'walking_stick', 'weevil',
    'isopod', 'snail', 'slug', 'harvestman', 'flatworm', 'chiton', 'tailed_frog',
    'trilobite', 'nematode', 'conch', 'sea_urchin', 'brain_coral',
    # Organic macro surface textures seen on genuine close-up leaves
    'velvet', 'handkerchief', 'swab', 'bonnet', 'dishrag', 'bath_towel',
    'wool', 'knot', 'coil', 'jigsaw_puzzle', 'doormat', 'paper_towel',
    'mosquito_net', 'stole', 'poncho', 'mitten', 'hair_slide', 'Band_Aid',
    'pencil_box', 'pencil_sharpener', 'wooden_spoon', 'ocarina', 'electric_ray',
    'quill', 'meerkat'
}

# Unambiguous non-plant keywords (characters, fantasy art, cinematic costumes, weapons, landscapes, etc.)
EXPLICIT_NON_PLANT_SUBSTRINGS = [
    # Fictional characters, costumes, armor, video games, cinema
    'cloak', 'cape', 'suit', 'costume', 'gown', 'robe', 'trench_coat',
    'academic_gown', 'military_uniform', 'breastplate', 'cuirass', 'armor',
    'crash_helmet', 'football_helmet', 'helmet', 'gasmask', 'mask', 'ski_mask',
    'scabbard', 'shield', 'sword', 'weapon', 'rifle', 'gun', 'cannon', 'missile',
    'stage', 'theatre', 'totem_pole', 'statue', 'sculpture', 'comic_book',
    'book_jacket', 'poster', 'alp', 'cliff', 'valley', 'promontory', 'volcano',
    'lakeside', 'sandbar', 'seashore', 'monastery', 'castle', 'church', 'prison',
    # Vehicles & Transportation
    'car', 'automobile', 'vehicle', 'truck', 'bus', 'van', 'jeep', 'wagon',
    'motorcycle', 'moped', 'bicycle', 'bike', 'plane', 'boat', 'train', 'wheel', 'tire',
    # Electronics, Digital Media, Office
    'computer', 'laptop', 'screen', 'monitor', 'cellphone', 'camera', 'television',
    'web_site', 'envelope', 'binder', 'rule', 'scoreboard', 'menu', 'packet', 'carton', 'abacus',
    # Animals & Pets
    'dog', 'cat', 'hound', 'terrier', 'horse', 'cow', 'bull', 'sheep', 'pig', 'elephant', 'bear', 'lion', 'tiger',
    # Furniture & Domestic
    'chair', 'sofa', 'bed', 'table', 'desk', 'wardrobe', 'toilet', 'refrigerator'
]

_mb_interp = None
_mb_in_idx = None
_mb_out_idx = None


def get_mobilenet_tflite():
    """Initializes or returns cached lightweight MobileNetV2 TFLite engine (~25ms)."""
    global _mb_interp, _mb_in_idx, _mb_out_idx
    if _mb_interp is None and MOBILENET_TFLITE_FILE.exists():
        try:
            interp = tf.lite.Interpreter(model_path=str(MOBILENET_TFLITE_FILE), num_threads=4)
            interp.allocate_tensors()
            _mb_interp = interp
            _mb_in_idx = interp.get_input_details()[0]["index"]
            _mb_out_idx = interp.get_output_details()[0]["index"]
        except Exception:
            _mb_interp = None
    return _mb_interp, _mb_in_idx, _mb_out_idx


def validate_image_file(file_bytes_or_stream, max_size_mb=20):
    """
    Validates file headers, format, dimensions, and image integrity.
    Supports JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF.
    """
    if isinstance(file_bytes_or_stream, bytes):
        if len(file_bytes_or_stream) > max_size_mb * 1024 * 1024:
            return False, f"File exceeds maximum size limit of {max_size_mb}MB.", None
        stream = io.BytesIO(file_bytes_or_stream)
    else:
        stream = file_bytes_or_stream

    try:
        img = Image.open(stream)
        img.verify()
    except Exception as e:
        return False, f"Invalid or corrupted image file: {str(e)}", None

    if isinstance(file_bytes_or_stream, bytes):
        stream = io.BytesIO(file_bytes_or_stream)
    else:
        stream.seek(0)

    try:
        img = Image.open(stream)
        fmt = (img.format or "").upper()
        allowed_formats = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF", "TIF", "GIF"}
        if fmt not in allowed_formats:
            return False, f"Unsupported format '{fmt}'. Please upload JPEG, PNG, WEBP, BMP, TIFF, or GIF.", None

        if getattr(img, "is_animated", False):
            img.seek(0)

        width, height = img.size
        if width < 32 or height < 32:
            return False, "Image dimensions too small (minimum 32x32 pixels required).", None
        if width > 10000 or height > 10000:
            return False, "Image dimensions exceed 10000x10000 pixels limit.", None

        img_rgb = img.convert("RGB")
        return True, "Valid image file", img_rgb
    except Exception as e:
        return False, f"Error processing image: {str(e)}", None


def analyze_foliage_chromaticity(pil_img):
    """Computes foliage color distribution and biological chlorophyll indices."""
    img_thumb = pil_img.resize((100, 100))
    arr = np.array(img_thumb, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Green dominance (g significantly exceeding r and b)
    green_dominant_ratio = float(np.mean((g > r + 10) & (g > b + 10)))

    # Chlorophyll presence (g > b + 6 with healthy luminance & chroma)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chlorophyll_ratio = float(np.mean((g > b + 6) & (g > 35) & ((max_c - min_c) > 20)))

    # Yellowing / chlorosis index (g > b + 8 and r > b + 8)
    yellow_ratio = float(np.mean((g > b + 8) & (r > b + 8) & (g > 35) & (r > 35) & ((max_c - min_c) > 20)))

    # Brown / necrotic lesion index (r > b + 15, moderate brightness)
    brown_ratio = float(np.mean((r > b + 15) & (g > b + 5) & (r > 40) & (r < 235) & ((max_c - min_c) > 25)))

    # Plant-like color fraction
    plant_color_fraction = green_dominant_ratio + 0.6 * chlorophyll_ratio + 0.3 * yellow_ratio + 0.3 * brown_ratio
    return {
        "green_dominant": green_dominant_ratio,
        "chlorophyll": chlorophyll_ratio,
        "yellow": yellow_ratio,
        "brown": brown_ratio,
        "plant_fraction": plant_color_fraction
    }


def is_valid_plant_leaf(pil_img):
    """
    Validates whether the image depicts an actual plant or leaf.
    Strictly rejects:
    - Fictional art, video game graphics (DOOM, fantasy characters), wallpapers, posters
    - Charts, documents, tables, white/gray canvas schematics
    - Vehicles, animals, clothing, furniture, architectural landscapes
    Returns: (is_plant: bool, reason: str, confidence_score: float)
    """
    arr_orig = np.array(pil_img.convert("RGB"), dtype=np.float32)
    r = arr_orig[:, :, 0]
    g = arr_orig[:, :, 1]
    b = arr_orig[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)

    # 1. Dark Shadow / Cinematic Scenery Check (detects DOOM, night scenes, dark fantasy art)
    dark_ratio = float(np.mean((r < 30) & (g < 30) & (b < 30)))
    if dark_ratio > 0.20:
        return False, f"Cinematic artwork or dark scenery detected ({dark_ratio*100:.1f}% shadow). Please upload a well-lit crop leaf.", 0.0

    # 2. White / Document / Chart Canvas Check (detects plots, tables, documents)
    white_ratio = float(np.mean((min_c > 220) & ((max_c - min_c) < 30)))
    chlorophyll = float(np.mean((g > b + 6) & (g > 35) & ((max_c - min_c) > 20)))
    green_dominant = float(np.mean((g > r + 10) & (g > b + 10)))
    if white_ratio > 0.25 and (chlorophyll < 0.12 and green_dominant < 0.10):
        return False, "Chart, diagram, or document canvas detected. Please upload an actual plant leaf photo.", 0.0

    # 3. Monochrome / Grayscale Check
    gray_ratio = float(np.mean((max_c - min_c) < 18))
    if gray_ratio > 0.85:
        return False, f"Monochrome or non-organic grayscale detected ({gray_ratio*100:.1f}%).", 0.0

    # 4. MobileNet Semantic Evaluation & Whitelist Verification
    interp, in_idx, out_idx = get_mobilenet_tflite()
    if interp is not None:
        try:
            img_resized = pil_img.convert("RGB").resize((224, 224))
            mob_arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.array(img_resized, dtype=np.float32), 0))
            interp.set_tensor(in_idx, mob_arr)
            interp.invoke()
            preds = decode_predictions(interp.get_tensor(out_idx), top=5)[0]

            top1_lbl = preds[0][1]
            top1_score = float(preds[0][2])

            # A. Check if Top-1 is explicitly non-plant (e.g. cloak, armor, cliff, car, comic_book)
            for non_plant_word in EXPLICIT_NON_PLANT_SUBSTRINGS:
                if non_plant_word in top1_lbl.lower():
                    clean_lbl = top1_lbl.replace("_", " ")
                    return False, f"Invalid Image: Please upload a valid plant or leaf image. (Detected non-plant object: {clean_lbl})", 0.0

            # B. Check top-3 predictions for strong explicit non-plant categories
            for _, l, s in preds[:3]:
                score = float(s)
                for non_plant_word in EXPLICIT_NON_PLANT_SUBSTRINGS:
                    if non_plant_word in l.lower() and score > 0.10:
                        clean_lbl = l.replace("_", " ")
                        return False, f"Invalid Image: Please upload a valid plant or leaf image. (Detected non-plant object: {clean_lbl})", 0.0

            # C. Whitelist enforcement: If Top-1 is NOT in allowed plant leaf categories with notable confidence
            if top1_lbl not in ALLOWED_LEAF_TOP1_CLASSES and top1_score > 0.10:
                clean_lbl = top1_lbl.replace("_", " ")
                return False, f"Invalid Image: Please upload a valid plant or leaf image. (Identified non-foliage object: {clean_lbl})", 0.0

        except Exception:
            pass

    # 5. Biological Foliage Chromaticity Verification
    chroma = analyze_foliage_chromaticity(pil_img)
    if chroma["plant_fraction"] < 0.035:
        return False, "Invalid Image: Image lacks botanical chlorophyll or foliage coloration.", chroma["plant_fraction"]

    return True, "Plant leaf visual characteristics confirmed.", chroma["plant_fraction"]


def assess_prediction_uncertainty(probabilities, min_confidence=0.40, max_entropy=0.75, min_margin=0.08):
    """
    Evaluates prediction uncertainty to guard against out-of-distribution or non-crop objects.
    Calculates normalized Shannon entropy and top-1 vs top-2 confidence margin.
    Returns: (is_confident: bool, uncertainty_message: str or None)
    """
    probs = np.array(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / np.sum(probs)

    num_classes = len(probs)
    entropy = -np.sum(probs * np.log2(probs)) / np.log2(num_classes)

    sorted_indices = np.argsort(probs)[::-1]
    top1_conf = float(probs[sorted_indices[0]])
    top2_conf = float(probs[sorted_indices[1]]) if num_classes > 1 else 0.0
    margin = top1_conf - top2_conf

    if top1_conf < min_confidence or entropy > max_entropy or margin < min_margin:
        return False, "Unable to identify this plant or disease reliably. Please upload a clear image of a supported crop leaf."

    return True, None
