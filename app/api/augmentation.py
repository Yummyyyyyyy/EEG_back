"""
Augmentation API endpoints
"""
import logging

from fastapi import APIRouter, HTTPException

from app.models.augmentation import AugmentationRequest, AugmentationResponse
from app.services.augmentation_service import augmentation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/augmentation", tags=["Augmentation"])


@router.post("/generate", response_model=AugmentationResponse)
async def generate_augmentation(payload: AugmentationRequest) -> AugmentationResponse:
    """
    Generate augmented EEG data using pre-generated samples.
    Currently supports the TCN method.
    """
    try:
        data = augmentation_service.generate(payload)

        return AugmentationResponse(
            success=True,
            data=data
        )

    except FileNotFoundError as exc:
        logger.error("Augmentation data not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        logger.error("Invalid augmentation request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except KeyError as exc:
        logger.error("Augmentation configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to generate augmentation data: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate augmentation data") from exc
