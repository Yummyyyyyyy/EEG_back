"""
Classification result models
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class MethodPrediction(BaseModel):
    """Single method's classification prediction"""
    method: str = Field(..., description="Augmentation method name")
    methodName: str = Field(..., description="Display name for frontend")
    predicted: str = Field(..., description="Predicted motion type")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    correct: bool = Field(..., description="Whether prediction matches true label")


class ClassificationResult(BaseModel):
    """Classification results for a trial"""
    trialId: str = Field(..., description="Trial ID")
    trueLabel: str = Field(..., description="True motion type")
    predictions: List[MethodPrediction] = Field(..., description="Predictions from different methods")
    numTrialsAggregated: int = Field(..., description="Number of trials aggregated")


class ClassificationResponse(BaseModel):
    """API response for classification query"""
    success: bool = True
    data: Optional[ClassificationResult] = None
    message: Optional[str] = None
