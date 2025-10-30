"""
Augmentation API endpoints
"""
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.augmentation import AugmentationRequest, AugmentationResponse, DownloadRequest
from app.services.augmentation_service import augmentation_service
from app.services.download_service import download_service

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


@router.post("/download")
async def download_augmentation(payload: DownloadRequest) -> StreamingResponse:
    """
    Download augmentation data as NPZ or CSV file.

    Args:
        payload: Download request with motion type, method, num samples, and file type

    Returns:
        StreamingResponse with the file content
    """
    try:
        # Load and process data
        trials, labels = download_service.load_augmentation_data(
            motion_type=payload.motionType,
            method=payload.method,
            num_samples=payload.numSamples
        )

        # Generate file based on requested type
        if payload.fileType == "npz":
            content = download_service.generate_npz(
                trials=trials,
                labels=labels,
                motion_type=payload.motionType,
                method=payload.method
            )
            media_type = "application/octet-stream"
            filename = f"{payload.method}_{payload.motionType}_{payload.numSamples}.npz"

            return StreamingResponse(
                iter([content]),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        elif payload.fileType == "csv":
            content = download_service.generate_csv(
                trials=trials,
                labels=labels,
                motion_type=payload.motionType
            )
            media_type = "text/csv"
            filename = f"{payload.method}_{payload.motionType}_{payload.numSamples}.csv"

            return StreamingResponse(
                iter([content]),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        else:
            raise ValueError(f"Unsupported file type: {payload.fileType}")

    except FileNotFoundError as exc:
        logger.error("Augmentation data not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        logger.error("Invalid download request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to download augmentation data: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to download augmentation data"
        ) from exc
