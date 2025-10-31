"""
Classification API endpoints
"""
import logging

from fastapi import APIRouter, HTTPException, Query

from app.models.classification import ClassificationResponse, ClassificationResult
from app.services.classification_service import classification_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classification", tags=["Classification"])


@router.get("/{trial_id}", response_model=ClassificationResponse)
async def get_classification(
    trial_id: str,
    motionType: str = Query(..., description="Motion type: left, right, foot, tongue"),
    numTrials: int = Query(10, ge=1, le=50, description="Number of trials to aggregate")
) -> ClassificationResponse:
    """
    Get EEGNet classification predictions for a trial

    This endpoint:
    1. Randomly selects N trials with the same motion type (including the requested trial)
    2. Aggregates predictions: finds the most predicted class (mode)
    3. Returns the maximum confidence for that predicted class

    Args:
        trial_id: Trial ID (e.g., "A01T001")
        motionType: Motion type to filter by (left, right, foot, tongue)
        numTrials: Number of trials to aggregate (default 10)

    Returns:
        Classification results from 5 augmentation methods
    """
    try:
        logger.info(
            f"Getting classification for trial_id={trial_id}, "
            f"motionType={motionType}, numTrials={numTrials}"
        )

        # Validate trial_id format
        if not trial_id or len(trial_id) < 7:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trial_id format: {trial_id}"
            )

        # Get predictions from service
        predictions = classification_service.get_classification_for_trial(
            trial_id=trial_id,
            motion_type=motionType,
            num_trials=numTrials
        )

        if not predictions:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for trial {trial_id}"
            )

        # Create result
        result = ClassificationResult(
            trialId=trial_id,
            trueLabel=motionType,
            predictions=predictions,
            numTrialsAggregated=numTrials
        )

        return ClassificationResponse(
            success=True,
            data=result
        )

    except ValueError as exc:
        logger.error(f"Invalid request parameters: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except FileNotFoundError as exc:
        logger.error(f"Prediction data not found: {exc}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"Failed to get classification: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to get classification results"
        ) from exc
