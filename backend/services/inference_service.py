"""
Inference & Agricultural Intelligence Service
Encapsulates model caching, prediction invocation, and metrics retrieval.
"""

import os
import json
from pathlib import Path
from deep_learning.predict import predict_image, load_inference_assets
from deep_learning.leaf_validator import validate_image_file, is_valid_plant_leaf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
INFO_FILE = DATA_DIR / "disease_information.json"
CLASS_FILE = MODELS_DIR / "class_names.json"


class InferenceService:
    @staticmethod
    def initialize():
        """Pre-warm model during FastAPI startup to ensure instant first prediction."""
        try:
            load_inference_assets()
            print("Inference service initialized and model pre-warmed.")
        except Exception as e:
            print(f"Notice: Model not pre-warmed (will load on demand or post-training): {e}")

    @staticmethod
    def predict(image_bytes: bytes, crop_filter: str = None) -> dict:
        return predict_image(image_bytes, crop_filter=crop_filter)

    @staticmethod
    def validate_image_only(image_bytes: bytes) -> dict:
        is_valid_fmt, fmt_msg, pil_img = validate_image_file(image_bytes)
        if not is_valid_fmt:
            return {
                "is_valid_file": False,
                "is_plant_leaf": False,
                "status": "invalid_file",
                "message": fmt_msg,
                "confidence_score": 0.0
            }

        is_plant, plant_msg, score = is_valid_plant_leaf(pil_img)
        return {
            "is_valid_file": True,
            "is_plant_leaf": is_plant,
            "status": "valid_plant" if is_plant else "invalid_non_plant",
            "message": plant_msg,
            "confidence_score": score
        }

    @staticmethod
    def get_classes():
        if CLASS_FILE.exists():
            with open(CLASS_FILE, "r") as f:
                return json.load(f)
        return []

    @staticmethod
    def get_crops():
        classes = InferenceService.get_classes()
        crop_dict = {}
        for cname in classes:
            parts = cname.split("___")
            crop = parts[0].replace("_", " ")
            disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            if crop not in crop_dict:
                crop_dict[crop] = []
            crop_dict[crop].append({
                "class_id": cname,
                "disease": disease,
                "is_healthy": "healthy" in disease.lower()
            })
        return [{"crop": k, "diseases": v, "total_categories": len(v)} for k, v in sorted(crop_dict.items())]

    @staticmethod
    def get_diseases():
        if INFO_FILE.exists():
            with open(INFO_FILE, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def get_disease_by_id(disease_id: str):
        diseases = InferenceService.get_diseases()
        return diseases.get(disease_id)

    @staticmethod
    def get_model_info():
        arch_file = RESULTS_DIR / "model_architecture.json"
        config_file = RESULTS_DIR / "training_config.json"
        
        info = {
            "architecture": "ResNet-50 Transfer Learning",
            "weights": "Pretrained on ImageNet",
            "input_resolution": "224 x 224 x 3",
            "framework": "TensorFlow / Keras",
            "classification_head": "GlobalAveragePooling2D -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(Softmax)",
            "supported_classes": len(InferenceService.get_classes())
        }

        if arch_file.exists():
            with open(arch_file, "r") as f:
                info["architecture_details"] = json.load(f)
        if config_file.exists():
            with open(config_file, "r") as f:
                info["training_configuration"] = json.load(f)

        return info

    @staticmethod
    def get_metrics():
        metrics_file = RESULTS_DIR / "evaluation_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def get_performance_dashboard():
        """Aggregates all evaluation results, curves, and summary stats for frontend dashboard."""
        metrics = InferenceService.get_metrics()
        model_info = InferenceService.get_model_info()
        
        history_file = RESULTS_DIR / "training_history.csv"
        training_history = []
        if history_file.exists():
            import pandas as pd
            df = pd.read_csv(history_file)
            training_history = df.to_dict(orient="records")

        report_file = RESULTS_DIR / "classification_report.csv"
        classification_report_data = []
        if report_file.exists():
            import pandas as pd
            df = pd.read_csv(report_file)
            classification_report_data = df.to_dict(orient="records")

        dataset_summary_file = RESULTS_DIR / "dataset_summary.json"
        dataset_summary = {}
        if dataset_summary_file.exists():
            with open(dataset_summary_file, "r") as f:
                dataset_summary = json.load(f)

        return {
            "model_info": model_info,
            "metrics": metrics,
            "training_history": training_history,
            "classification_report": classification_report_data,
            "dataset_summary": dataset_summary
        }
