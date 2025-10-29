"""
Trial管理API接口
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from app.models.trial import TrialListResponse
from app.models.eeg import EEGDataResponse
from app.services.data_loader import data_loader
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trials", tags=["Trials"])


@router.get("", response_model=TrialListResponse)
async def get_trials(
    subject: Optional[int] = Query(None, ge=1, le=9, description="受试者编号 1-9"),
    motionType: Optional[str] = Query(None, description="运动类型: left, right, foot, tongue")
):
    """
    获取所有trials列表

    支持按受试者和运动类型筛选
    """
    try:
        logger.info(f"Getting trials: subject={subject}, motionType={motionType}")

        trials = data_loader.get_all_trials(
            subject=subject,
            motion_type=motionType
        )

        return TrialListResponse(
            success=True,
            data=trials,
            total=len(trials)
        )

    except Exception as e:
        logger.error(f"Error getting trials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{trial_id}/eeg-data", response_model=EEGDataResponse)
async def get_eeg_data(
    trial_id: str,
    removeEOG: bool = Query(True, description="是否移除EOG（已废弃，始终为true）"),
    extractMI: bool = Query(False, description="是否提取MI片段")
):
    """
    获取特定trial的EEG数据

    Args:
        trial_id: Trial ID，格式如 "A01T001"
        removeEOG: 是否移除EOG（参数保留，实际两个数据源都已移除EOG）
        extractMI: 是否使用MI提取后的数据
            - False: 使用processed数据（去除EOG，保留完整时序）
            - True: 使用cleaned数据（去除EOG + 提取MI段）

    Returns:
        包含original和processed两种数据的响应
        - original: processed数据（向前端兼容）
        - processed: 根据extractMI参数返回的数据
    """
    try:
        logger.info(f"Getting EEG data: trial_id={trial_id}, extractMI={extractMI}")

        # 验证trial_id格式
        if not trial_id or len(trial_id) < 7:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trial_id format: {trial_id}"
            )

        # 加载processed版本（作为original）
        original_data = data_loader.load_trial_data(
            trial_id=trial_id,
            extract_mi=False
        )

        # 加载processed或cleaned版本（根据extractMI参数）
        processed_data = data_loader.load_trial_data(
            trial_id=trial_id,
            extract_mi=extractMI
        )

        return EEGDataResponse(
            success=True,
            data={
                "original": original_data,
                "processed": processed_data
            }
        )

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        logger.error(f"Invalid parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error loading EEG data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))