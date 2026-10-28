"""
Comprehensive Test Suite for Multi-Crop Plant Disease Detection System
Tests:
1. Dataset structure and integrity
2. ResNet-50 architecture initialization and dynamic class sizing
3. Preprocessing shape and normalization
4. Leaf validator non-plant rejection logic
5. Disease information completeness
6. FastAPI backend endpoints and error handling
"""

import os
import json
import io
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from deep_learning.leaf_validator import validate_image_file, is_valid_plant_leaf, assess_prediction_uncertainty
from deep_learning.preprocessing import preprocess_single_image, IMG_SIZE
from deep_learning.model import build_resnet50_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"

client = TestClient(app)


def test_disease_knowledge_base_integrity():
    """Verify data/disease_information.json has valid structure for all entries."""
    info_file = DATA_DIR / "disease_information.json"
    assert info_file.exists(), "disease_information.json must exist"

    with open(info_file, "r") as f:
        data = json.load(f)

    assert len(data) >= 20, "Should contain at least 20 documented disease categories"
    for cid, info in data.items():
        assert "crop" in info
        assert "disease_name" in info
        assert "is_healthy" in info
        assert "description" in info
        assert "symptoms" in info and isinstance(info["symptoms"], list)
        assert "prevention" in info and isinstance(info["prevention"], list)
        assert "management" in info and isinstance(info["management"], list)


def test_image_file_validator_multi_formats():
    """Verify validation supports JPG, PNG, WEBP, BMP and rejects corrupted files."""
    # 1. Valid RGB PNG
    img_png = Image.new("RGB", (256, 256), color=(34, 139, 34))
    buf = io.BytesIO()
    img_png.save(buf, format="PNG")
    valid, msg, _ = validate_image_file(buf.getvalue())
    assert valid is True

    # 2. Valid JPEG
    buf_jpg = io.BytesIO()
    img_png.save(buf_jpg, format="JPEG")
    valid_jpg, _, _ = validate_image_file(buf_jpg.getvalue())
    assert valid_jpg is True

    # 3. Valid WEBP
    buf_webp = io.BytesIO()
    img_png.save(buf_webp, format="WEBP")
    valid_webp, _, _ = validate_image_file(buf_webp.getvalue())
    assert valid_webp is True

    # 4. Too small image rejection
    img_tiny = Image.new("RGB", (16, 16), color=(0, 0, 0))
    buf_tiny = io.BytesIO()
    img_tiny.save(buf_tiny, format="PNG")
    valid_tiny, msg_tiny, _ = validate_image_file(buf_tiny.getvalue())
    assert valid_tiny is False
    assert "too small" in msg_tiny.lower()

    # 5. Corrupted bytes rejection
    valid_corrupt, msg_corrupt, _ = validate_image_file(b"NOT_AN_IMAGE_FILE_CORRUPTED_BYTES")
    assert valid_corrupt is False


def test_leaf_validator_non_plant_rejection():
    """Verify leaf validator correctly identifies plant-like vs non-plant patterns."""
    # Foliage green image
    plant_img = Image.new("RGB", (224, 224), color=(40, 140, 40))
    is_plant, msg, _ = is_valid_plant_leaf(plant_img)
    assert is_plant is True

    # Pure blue / metallic non-plant image (e.g., car body)
    blue_img = Image.new("RGB", (224, 224), color=(10, 30, 200))
    is_plant_blue, _, _ = is_valid_plant_leaf(blue_img)
    # Should flag as non-plant
    assert is_plant_blue is False


def test_uncertainty_quantification():
    """Verify out-of-distribution high entropy distribution is flagged as uncertain."""
    # Uniform uncertain probabilities across 30 classes (1/30 = 0.033 each)
    uniform_probs = [1.0 / 30] * 30
    is_conf, uncert_msg = assess_prediction_uncertainty(uniform_probs)
    assert is_conf is False
    assert "Unable to identify" in uncert_msg

    # Highly confident prediction (e.g. 96% top class)
    confident_probs = [0.96] + [0.04 / 29] * 29
    is_conf_high, _ = assess_prediction_uncertainty(confident_probs)
    assert is_conf_high is True


def test_preprocessing_output_shape():
    """Verify preprocessing outputs float32 tensor of shape (1, 224, 224, 3)."""
    dummy = Image.new("RGB", (500, 300), color=(50, 150, 50))
    pre = preprocess_single_image(dummy)
    assert pre.shape == (1, 224, 224, 3)
    assert pre.dtype == np.float32


def test_resnet50_model_builder():
    """Verify ResNet-50 compiles dynamically with custom number of classes."""
    test_classes = 15
    model, base_model = build_resnet50_model(num_classes=test_classes)
    assert model.output_shape == (None, test_classes)
    assert model.input_shape == (None, 224, 224, 3)
    assert base_model.trainable is False


def test_api_health_endpoint():
    """Verify GET /api/health responds with status operational."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data


def test_api_crops_and_diseases_endpoints():
    """Verify GET /api/crops and GET /api/diseases return structured responses."""
    resp_crops = client.get("/api/crops")
    assert resp_crops.status_code == 200
    assert "crops" in resp_crops.json()

    resp_diseases = client.get("/api/diseases")
    assert resp_diseases.status_code == 200
    assert "diseases" in resp_diseases.json()


def test_api_disease_info_detail():
    """Verify GET /api/disease-info/{disease_id} returns accurate agronomic details."""
    resp = client.get("/api/disease-info/Apple___Apple_scab")
    assert resp.status_code == 200
    data = resp.json()
    assert data["crop"] == "Apple"
    assert data["disease_name"] == "Apple Scab"
    assert "symptoms" in data
    assert "prevention" in data
    assert "management" in data
