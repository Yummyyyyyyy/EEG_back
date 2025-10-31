"""
Classification service for loading and aggregating EEGNet predictions

This service:
1. Loads prediction results from CSV files
2. Implements the aggregation logic: randomly select 10 trials with same label
3. Returns the most predicted class (mode) with its maximum confidence
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import pandas as pd

from app.models.classification import MethodPrediction


class ClassificationService:
    """
    Service for loading and aggregating EEGNet classification predictions
    """

    # Method configuration
    METHODS = {
        'orig': 'Original',
        'VAE': 'VAE',
        'TCN': 'TCN',
        'GAN': 'GAN',
        'Diffusion': 'Diffusion'
    }

    # Motion type mapping
    LABEL_TO_MOTION = {
        0: 'left',
        1: 'right',
        2: 'feet',
        3: 'tongue'
    }

    MOTION_TO_LABEL = {
        'left': 0,
        'right': 1,
        'feet': 2,
        'foot': 2,  # Alias
        'tongue': 3
    }

    def __init__(self) -> None:
        base_dir = Path(__file__).parent.parent.parent
        self._predictions_dir = base_dir / "app/data/predictions"
        self._mapping_file = base_dir / "app/data/trial_id_mapping.pkl"

        # Cache for prediction data
        self._predictions_cache: Dict[str, pd.DataFrame] = {}
        self._mapping: Optional[Dict] = None

        # Load mapping on initialization
        self._load_mapping()

    def _load_mapping(self) -> None:
        """Load trial ID mapping"""
        if not self._mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {self._mapping_file}")

        with open(self._mapping_file, 'rb') as f:
            self._mapping = pickle.load(f)

    def _load_predictions(self, method: str) -> pd.DataFrame:
        """
        Load prediction CSV for a specific method

        Args:
            method: Method name (orig, VAE, TCN, GAN, Diffusion)

        Returns:
            DataFrame with predictions
        """
        if method in self._predictions_cache:
            return self._predictions_cache[method]

        csv_path = self._predictions_dir / f"EEGNet_{method}_predictions.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        self._predictions_cache[method] = df

        return df

    def _get_trials_with_same_label(
        self,
        df: pd.DataFrame,
        true_label: int,
        include_trial_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get all trials with the same true label

        Args:
            df: Predictions DataFrame
            true_label: True label (0-3)
            include_trial_id: Ensure this trial_id is included

        Returns:
            Filtered DataFrame
        """
        # Filter by true label
        same_label_df = df[df['true_label'] == true_label].copy()

        # If specific trial_id requested, ensure it's included
        if include_trial_id:
            if include_trial_id not in same_label_df['trial_id'].values:
                # Trial not found with this label - this shouldn't happen
                raise ValueError(
                    f"Trial {include_trial_id} not found with label {true_label}"
                )

        return same_label_df

    def _select_random_trials(
        self,
        df: pd.DataFrame,
        num_trials: int,
        include_trial_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Randomly select trials, ensuring specific trial_id is included if provided

        Args:
            df: DataFrame to sample from
            num_trials: Number of trials to select
            include_trial_id: Ensure this trial_id is included

        Returns:
            Sampled DataFrame
        """
        # Remove NaN trial_ids (augmented data without real trial_id)
        df_valid = df[df['trial_id'].notna()].copy()

        if len(df_valid) == 0:
            raise ValueError("No valid trials with trial_id found")

        # If we need to include a specific trial
        if include_trial_id:
            # Get the specific trial
            specific_trial = df_valid[df_valid['trial_id'] == include_trial_id]

            if len(specific_trial) == 0:
                raise ValueError(f"Trial {include_trial_id} not found")

            # Get other trials
            other_trials = df_valid[df_valid['trial_id'] != include_trial_id]

            # Sample remaining trials
            num_to_sample = min(num_trials - 1, len(other_trials))

            if num_to_sample > 0:
                sampled_others = other_trials.sample(n=num_to_sample, random_state=None)
                result = pd.concat([specific_trial, sampled_others])
            else:
                result = specific_trial

        else:
            # Just random sample
            num_to_sample = min(num_trials, len(df_valid))
            result = df_valid.sample(n=num_to_sample, random_state=None)

        return result

    def _aggregate_predictions(self, sampled_df: pd.DataFrame) -> Dict:
        """
        Aggregate predictions: find mode (most common prediction) and its max confidence

        Args:
            sampled_df: DataFrame with sampled trials

        Returns:
            Dictionary with predicted_label and confidence
        """
        if len(sampled_df) == 0:
            raise ValueError("No trials to aggregate")

        # Count predictions (mode)
        prediction_counts = Counter(sampled_df['predicted_label'])
        most_common_label, _ = prediction_counts.most_common(1)[0]

        # Get maximum confidence for the most common prediction
        same_prediction = sampled_df[sampled_df['predicted_label'] == most_common_label]
        max_confidence = same_prediction['confidence'].max()

        return {
            'predicted_label': int(most_common_label),
            'confidence': float(max_confidence)
        }

    def get_classification_for_trial(
        self,
        trial_id: str,
        motion_type: str,
        num_trials: int = 10
    ) -> List[MethodPrediction]:
        """
        Get classification predictions for a trial

        Args:
            trial_id: Trial ID (e.g., "A01T001")
            motion_type: Motion type to filter by
            num_trials: Number of trials to aggregate

        Returns:
            List of MethodPrediction objects
        """
        # Convert motion type to label
        if motion_type not in self.MOTION_TO_LABEL:
            raise ValueError(f"Invalid motion type: {motion_type}")

        true_label = self.MOTION_TO_LABEL[motion_type]
        true_motion = self.LABEL_TO_MOTION[true_label]

        predictions = []

        # Process each method
        for method_key, method_name in self.METHODS.items():
            try:
                # Load predictions for this method
                df = self._load_predictions(method_key)

                # Get trials with same label
                same_label_df = self._get_trials_with_same_label(
                    df, true_label, include_trial_id=trial_id
                )

                # Randomly select trials (including the requested trial_id)
                sampled_df = self._select_random_trials(
                    same_label_df, num_trials, include_trial_id=trial_id
                )

                # Aggregate predictions
                result = self._aggregate_predictions(sampled_df)

                # Convert to motion type
                predicted_label = result['predicted_label']
                predicted_motion = self.LABEL_TO_MOTION[predicted_label]
                confidence = result['confidence']

                # Check if correct
                correct = (predicted_label == true_label)

                # Create prediction object
                prediction = MethodPrediction(
                    method=method_key.lower(),
                    methodName=method_name,
                    predicted=predicted_motion,
                    confidence=confidence,
                    correct=correct
                )

                predictions.append(prediction)

            except Exception as e:
                # Log error but continue with other methods
                print(f"Error processing method {method_key}: {e}")
                continue

        return predictions


# Create global singleton
classification_service = ClassificationService()
