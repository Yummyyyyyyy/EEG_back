"""
Trial数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional


class Trial(BaseModel):
    """
    Trial（试次）数据模型
    """
    id: str = Field(..., description="Trial ID，格式如 'A01T001'")
    subject: int = Field(..., ge=1, le=9, description="受试者编号 1-9")
    trialIndex: int = Field(..., ge=1, le=288, description="Trial索引 1-288")
    motionType: str = Field(..., description="运动类型: left, right, foot, tongue")
    label: int = Field(..., ge=0, le=3, description="运动类型标签 0-3")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "A01T001",
                "subject": 1,
                "trialIndex": 1,
                "motionType": "left",
                "label": 0
            }
        }


class TrialListResponse(BaseModel):
    """
    Trial列表响应模型
    """
    success: bool = True
    data: list[Trial]
    total: int

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [
                    {
                        "id": "A01T001",
                        "subject": 1,
                        "trialIndex": 1,
                        "motionType": "left",
                        "label": 0
                    }
                ],
                "total": 288
            }
        }