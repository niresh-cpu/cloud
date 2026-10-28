"""
Pydantic schemas for structured API responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_id: str
    crop: str
    disease: str
    confidence: float
    confidence_percentage: str


class PredictionResponse(BaseModel):
    success: bool
    status: str
    crop: Optional[str] = None
    disease: Optional[str] = None
    class_id: Optional[str] = None
    is_healthy: Optional[bool] = None
    status_label: Optional[str] = None
    confidence: Optional[float] = None
    confidence_percentage: Optional[str] = None
    top_predictions: Optional[List[TopPrediction]] = None
    pathogen_type: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    symptoms: Optional[List[str]] = None
    prevention: Optional[List[str]] = None
    management: Optional[List[str]] = None
    inference_time_ms: float
    validation_status: Optional[str] = None
    message: Optional[str] = None
    details: Optional[str] = None
    provisional_class: Optional[str] = None
    crop_filter: Optional[str] = None
    crop_breakdown: Optional[List[Dict[str, Any]]] = None


class ImageValidationResponse(BaseModel):
    is_valid_file: bool
    is_plant_leaf: bool
    status: str
    message: str
    confidence_score: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_classes: int
    version: str


class DiseaseInfoResponse(BaseModel):
    disease_id: str
    crop: str
    disease_name: str
    is_healthy: bool
    pathogen_type: str
    description: str
    symptoms: List[str]
    prevention: List[str]
    management: List[str]
    severity: str
