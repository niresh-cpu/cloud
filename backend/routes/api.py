"""
FastAPI Router for Real-Time Plant Disease Detection System
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.services.inference_service import InferenceService
from backend.schemas.response_models import (
    PredictionResponse,
    ImageValidationResponse,
    HealthResponse
)

router = APIRouter(prefix="/api", tags=["Agricultural AI API"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


@router.get("/health", response_model=HealthResponse)
def health_check():
    """System health and operational state verification."""
    classes = InferenceService.get_classes()
    model_exists = (MODELS_DIR / "crop_disease_resnet50.keras").exists()
    return {
        "status": "operational",
        "model_loaded": model_exists,
        "total_classes": len(classes),
        "version": "1.0.0"
    }


from starlette.concurrency import run_in_threadpool

@router.post("/predict", response_model=PredictionResponse)
async def predict_plant_disease(
    file: UploadFile = File(...),
    crop: str = Form(None)
):
    """
    Primary prediction endpoint:
    Accepts multipart image upload, validates leaf integrity, and returns ResNet-50 predictions.
    Optional crop parameter restricts/focuses diagnosis to a specific crop.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        result = await run_in_threadpool(InferenceService.predict, image_bytes, crop)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")


@router.post("/validate-image", response_model=ImageValidationResponse)
async def validate_image(file: UploadFile = File(...)):
    """Validates image integrity and checks if input contains valid plant foliage."""
    try:
        image_bytes = await file.read()
        return await run_in_threadpool(InferenceService.validate_image_only, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.get("/classes")
def get_supported_classes():
    """Returns list of all classes supported by the trained ResNet-50 model."""
    classes = InferenceService.get_classes()
    return {"total": len(classes), "classes": classes}


@router.get("/crops")
def get_supported_crops():
    """Returns all supported crop species and their associated disease categories."""
    crops = InferenceService.get_crops()
    return {"total_crops": len(crops), "crops": crops}


@router.get("/diseases")
def get_all_diseases():
    """Returns knowledge database of diseases and prevention protocols."""
    diseases = InferenceService.get_diseases()
    return {"total": len(diseases), "diseases": diseases}


@router.get("/disease-info/{disease_id}")
def get_disease_detail(disease_id: str):
    """Retrieves agronomic description and management advice for specific disease."""
    info = InferenceService.get_disease_by_id(disease_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Disease record '{disease_id}' not found.")
    return {"disease_id": disease_id, **info}


@router.get("/model-info")
def get_model_information():
    """Returns ResNet-50 model architecture, parameters, and layer specifications."""
    return InferenceService.get_model_info()


@router.get("/model-metrics")
def get_model_metrics():
    """Returns test set accuracy, precision, recall, and F1-score."""
    metrics = InferenceService.get_metrics()
    if not metrics:
        return {"status": "Metrics pending evaluation. Please run evaluation script."}
    return metrics


@router.get("/performance")
def get_performance_dashboard_data():
    """Returns aggregated performance data including training curves and classification report."""
    return InferenceService.get_performance_dashboard()


@router.get("/charts/{chart_name}")
def get_performance_chart(chart_name: str):
    """Serves high-resolution evaluation chart figures."""
    allowed = {
        "class_distribution.png",
        "accuracy_curve.png",
        "loss_curve.png",
        "confusion_matrix.png",
        "class_f1_scores.png"
    }
    if chart_name not in allowed:
        raise HTTPException(status_code=404, detail="Requested chart not found.")
    chart_path = RESULTS_DIR / chart_name
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail=f"Chart '{chart_name}' has not been generated yet.")
    return FileResponse(chart_path, media_type="image/png")


@router.get("/samples")
def get_sample_test_images():
    """Returns a list of curated sample test images for instant one-click UI demonstration."""
    DATASETS_DIR = BASE_DIR / "datasets"
    TEST_DIR = DATASETS_DIR / "test"
    NON_PLANT_DIR = DATASETS_DIR / "non_plant_test"

    samples = [
        {
            "id": "apple_black_rot",
            "title": "Apple: Black Rot",
            "crop": "Apple",
            "type": "Diseased Leaf",
            "folder": "test/Apple___Black_rot",
            "description": "Foliar circular frog-eye spots"
        },
        {
            "id": "tomato_healthy",
            "title": "Tomato: Healthy",
            "crop": "Tomato",
            "type": "Healthy Leaf",
            "folder": "test/Tomato___healthy",
            "description": "Clean foliage, no lesions"
        },
        {
            "id": "potato_late_blight",
            "title": "Potato: Late Blight",
            "crop": "Potato",
            "type": "Diseased Leaf",
            "folder": "test/Potato___Late_blight",
            "description": "Water-soaked dark lesions"
        },
        {
            "id": "rice_brown_spot",
            "title": "Rice: Brown Spot",
            "crop": "Rice",
            "type": "Diseased Leaf",
            "folder": "test/Rice___Brown_Spot",
            "description": "Oval brown sesame lesions"
        },
        {
            "id": "corn_rust",
            "title": "Corn: Common Rust",
            "crop": "Corn",
            "type": "Diseased Leaf",
            "folder": "test/Corn_(maize)___Common_rust_",
            "description": "Cinnamon-brown pustules"
        },
        {
            "id": "non_plant_guard",
            "title": "Non-Plant (Vehicle)",
            "crop": "Non-Plant",
            "type": "Rejection Guard Test",
            "folder": "non_plant_test",
            "description": "Tests foliage rejection guard"
        }
    ]

    result = []
    for s in samples:
        target_dir = DATASETS_DIR / s["folder"]
        if target_dir.exists():
            files = list(target_dir.glob("*.[jJ][pP][gG]")) + list(target_dir.glob("*.[pP][nN][gG]"))
            if files:
                first_file = files[0].name
                result.append({
                    **s,
                    "filename": first_file,
                    "image_url": f"/api/sample-file?folder={s['folder']}&filename={first_file}"
                })

    return {"samples": result}


@router.get("/sample-file")
def get_sample_file(folder: str, filename: str):
    """Serves the actual sample image bytes."""
    DATASETS_DIR = BASE_DIR / "datasets"
    # Basic path traversal protection
    clean_folder = folder.replace("..", "").strip("/\\")
    clean_filename = Path(filename).name
    file_path = DATASETS_DIR / clean_folder / clean_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image file not found.")

    suffix = file_path.suffix.lower()
    media_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    return FileResponse(file_path, media_type=media_type)
