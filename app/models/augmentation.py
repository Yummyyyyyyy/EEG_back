"""
Augmentation data models
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, conlist

from app.models.eeg import EEGData


class AugmentedChannels(BaseModel):
    """
    Aggregated augmented EEG signals for the target channels
    """
    Fz: List[Optional[float]] = Field(..., description="Fz channel averaged signal")
    C3: List[Optional[float]] = Field(..., description="C3 channel averaged signal")
    Cz: List[Optional[float]] = Field(..., description="Cz channel averaged signal")
    C4: List[Optional[float]] = Field(..., description="C4 channel averaged signal")
    Pz: List[Optional[float]] = Field(..., description="Pz channel averaged signal")


class AugmentedData(BaseModel):
    """
    Single augmentation result for a method
    """
    method: str = Field(..., description="Augmentation method identifier, e.g. 'tcn'")
    channels: AugmentedChannels = Field(..., description="Averaged channel signals")


class AugmentationResponse(BaseModel):
    """
    Augmentation response wrapper
    """
    success: bool = True
    data: Dict[str, AugmentedData] = Field(default_factory=dict)


class AugmentationRequest(BaseModel):
    """
    Augmentation generation request body
    """
    trialId: str = Field(..., description="Trial identifier")
    methods: conlist(str, min_length=1) = Field(
        ...,
        description="List of augmentation methods to generate"
    )
    eegData: EEGData = Field(..., description="Processed EEG data for the current trial")
    count: Optional[int] = Field(
        10,
        ge=1,
        le=1000,
        description="Number of augmented samples to aggregate per method"
    )
