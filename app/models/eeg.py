"""
EEG数据模型
"""
from pydantic import BaseModel, Field
from typing import Dict, List
from app.models.trial import Trial


class EEGChannels(BaseModel):
    """
    5个通道的EEG数据
    """
    Fz: List[float] = Field(..., description="Fz通道数据")
    C3: List[float] = Field(..., description="C3通道数据")
    Cz: List[float] = Field(..., description="Cz通道数据")
    C4: List[float] = Field(..., description="C4通道数据")
    Pz: List[float] = Field(..., description="Pz通道数据")


class EEGData(BaseModel):
    """
    单个Trial的EEG数据
    """
    channels: EEGChannels = Field(..., description="5个通道的数据")
    labels: List[float] = Field(..., description="时间标签（秒）")
    samplingRate: int = Field(250, description="采样率 Hz")
    trialInfo: Trial = Field(..., description="Trial信息")

    class Config:
        json_schema_extra = {
            "example": {
                "channels": {
                    "Fz": [0.0001, 0.0002, 0.0001],
                    "C3": [0.0001, 0.0002, 0.0001],
                    "Cz": [0.0001, 0.0002, 0.0001],
                    "C4": [0.0001, 0.0002, 0.0001],
                    "Pz": [0.0001, 0.0002, 0.0001]
                },
                "labels": [0.0, 0.004, 0.008],
                "samplingRate": 250,
                "trialInfo": {
                    "id": "A01T001",
                    "subject": 1,
                    "trialIndex": 1,
                    "motionType": "left",
                    "label": 0
                }
            }
        }


class EEGDataResponse(BaseModel):
    """
    EEG数据响应模型
    """
    success: bool = True
    data: Dict[str, EEGData] = Field(
        ...,
        description="包含original和processed两种数据"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "original": {
                        "channels": {"Fz": [], "C3": [], "Cz": [], "C4": [], "Pz": []},
                        "labels": [],
                        "samplingRate": 250,
                        "trialInfo": {
                            "id": "A01T001",
                            "subject": 1,
                            "trialIndex": 1,
                            "motionType": "left",
                            "label": 0
                        }
                    },
                    "processed": {
                        "channels": {"Fz": [], "C3": [], "Cz": [], "C4": [], "Pz": []},
                        "labels": [],
                        "samplingRate": 250,
                        "trialInfo": {
                            "id": "A01T001",
                            "subject": 1,
                            "trialIndex": 1,
                            "motionType": "left",
                            "label": 0
                        }
                    }
                }
            }
        }