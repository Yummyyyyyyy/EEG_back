"""
Download service for augmentation data.
Handles data sampling, padding, and file generation (NPZ and CSV).
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.config import settings


class DownloadService:
    """
    Service for downloading augmentation data in NPZ or CSV format.
    """

    MOTION_TYPE_MAP = {
        "left": 0,
        "right": 1,
        "foot": 2,
        "feet": 2,
        "tongue": 3,
    }

    def __init__(self) -> None:
        base_dir = Path(__file__).parent.parent.parent
        self._tcn_dir = base_dir / "app/data/augmentation/tcn"
        self._gan_file = base_dir / "app/data/augmentation/gan/generated_trials.npz"
        self._vae_file = base_dir / "app/data/augmentation/vae/baseline_synthetic_generated_batch.npz"
        self._diffusion_dir = base_dir / "app/data/augmentation/diffusion"

        # File mapping for different methods
        self._tcn_file_map = {
            "left": "left_1000.npz",
            "right": "right_1000.npz",
            "foot": "feet_1000.npz",
            "feet": "feet_1000.npz",
            "tongue": "tongue_1000.npz",
        }

        self._diffusion_file_map = {
            "left": "samples_class0_left.npz",
            "right": "samples_class1_right.npz",
            "foot": "samples_class2_feet.npz",
            "feet": "samples_class2_feet.npz",
            "tongue": "samples_class3_tongue.npz",
        }

    def load_augmentation_data(
        self,
        motion_type: str,
        method: str,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load augmentation data for the specified motion type and method.
        Returns (trials, labels) with sampling or padding applied.

        Args:
            motion_type: MI type (left, right, foot, tongue)
            method: Augmentation method (tcn, gan, vae, diffusion)
            num_samples: Number of samples to return

        Returns:
            Tuple of (trials_array, labels_array)
            - trials_array: shape (num_samples, channels, time_points)
            - labels_array: shape (num_samples,)
        """
        method = method.lower()
        motion_type = motion_type.lower()

        if method == "tcn":
            trials, labels = self._load_tcn_data(motion_type)
        elif method == "gan":
            trials, labels = self._load_gan_data(motion_type)
        elif method == "vae":
            trials, labels = self._load_vae_data(motion_type)
        elif method == "diffusion":
            trials, labels = self._load_diffusion_data(motion_type)
        else:
            raise ValueError(f"Unsupported augmentation method: {method}")

        # Apply sampling or padding
        trials, labels = self._sample_or_pad(trials, labels, num_samples)

        return trials, labels

    def _load_tcn_data(self, motion_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load TCN augmentation data."""
        if motion_type not in self._tcn_file_map:
            raise ValueError(f"Unsupported motion type for TCN: {motion_type}")

        file_path = self._tcn_dir / self._tcn_file_map[motion_type]
        if not file_path.exists():
            raise FileNotFoundError(f"TCN data not found: {file_path}")

        with np.load(file_path) as data:
            trials = data["eeg_data"]  # Shape: (N, 22, 1000)
            labels = data.get("labels")

        # Filter by motion type if labels exist
        if labels is not None:
            target_label = self.MOTION_TYPE_MAP[motion_type]
            mask = labels == target_label
            trials = trials[mask]
            labels = labels[mask]
        else:
            # If no labels, create them based on motion type
            target_label = self.MOTION_TYPE_MAP[motion_type]
            labels = np.full(trials.shape[0], target_label, dtype=np.int32)

        return trials, labels

    def _load_gan_data(self, motion_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load GAN augmentation data."""
        if not self._gan_file.exists():
            raise FileNotFoundError(f"GAN data not found: {self._gan_file}")

        with np.load(self._gan_file) as data:
            trials = data["trials"]  # Shape: (2000, 22, 500)
            labels = data["labels"]

        # Filter by motion type
        target_label = self.MOTION_TYPE_MAP[motion_type]
        mask = labels == target_label
        trials = trials[mask]
        labels = labels[mask]

        return trials, labels

    def _load_vae_data(self, motion_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load VAE augmentation data."""
        if not self._vae_file.exists():
            raise FileNotFoundError(f"VAE data not found: {self._vae_file}")

        with np.load(self._vae_file) as data:
            trials = data["X_gen"]  # Shape: (2400, 22, 1000, 1)
            labels = data["y_gen"]

        # Remove last dimension if present
        if trials.ndim == 4 and trials.shape[-1] == 1:
            trials = trials[..., 0]

        # Filter by motion type
        target_label = self.MOTION_TYPE_MAP[motion_type]
        mask = labels == target_label
        trials = trials[mask]
        labels = labels[mask]

        return trials, labels

    def _load_diffusion_data(self, motion_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load Diffusion augmentation data."""
        if motion_type not in self._diffusion_file_map:
            raise ValueError(f"Unsupported motion type for Diffusion: {motion_type}")

        file_path = self._diffusion_dir / self._diffusion_file_map[motion_type]
        if not file_path.exists():
            raise FileNotFoundError(f"Diffusion data not found: {file_path}")

        with np.load(file_path) as data:
            trials = data["X"]  # Shape: (500, 22, 1000)
            labels = data.get("y")

        # Filter by motion type if labels exist
        if labels is not None:
            target_label = self.MOTION_TYPE_MAP[motion_type]
            mask = labels == target_label
            trials = trials[mask]
            labels = labels[mask]
        else:
            # If no labels, create them based on motion type
            target_label = self.MOTION_TYPE_MAP[motion_type]
            labels = np.full(trials.shape[0], target_label, dtype=np.int32)

        return trials, labels

    def _sample_or_pad(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample or pad data to match the requested number of samples.

        Args:
            trials: Input trials array
            labels: Input labels array
            num_samples: Target number of samples

        Returns:
            Tuple of (sampled/padded trials, sampled/padded labels)
        """
        available_samples = trials.shape[0]

        if available_samples == num_samples:
            return trials, labels

        if available_samples > num_samples:
            # Random sampling
            indices = random.sample(range(available_samples), num_samples)
            indices.sort()  # Keep chronological order
            return trials[indices], labels[indices]

        # Padding with zeros
        num_padding = num_samples - available_samples
        channels = trials.shape[1]
        time_points = trials.shape[2]

        # Create zero-padded arrays
        padded_trials = np.zeros((num_samples, channels, time_points), dtype=trials.dtype)
        padded_labels = np.zeros(num_samples, dtype=labels.dtype)

        # Copy existing data
        padded_trials[:available_samples] = trials
        padded_labels[:available_samples] = labels

        return padded_trials, padded_labels

    def generate_npz(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        motion_type: str,
        method: str
    ) -> bytes:
        """
        Generate NPZ file from trials and labels.

        Args:
            trials: Trials array
            labels: Labels array
            motion_type: MI type
            method: Augmentation method

        Returns:
            NPZ file content as bytes
        """
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            trials=trials,
            labels=labels,
            motion_type=motion_type,
            method=method,
            sfreq=settings.SAMPLING_RATE
        )
        buffer.seek(0)
        return buffer.read()

    def generate_csv(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        motion_type: str
    ) -> str:
        """
        Generate CSV file from trials and labels.

        CSV columns: motion_type, trial_index, time_point, eeg_signals

        Args:
            trials: Trials array (num_trials, channels, time_points)
            labels: Labels array (num_trials,)
            motion_type: MI type

        Returns:
            CSV file content as string
        """
        rows = []

        num_trials, num_channels, num_time_points = trials.shape

        for trial_idx in range(num_trials):
            for time_idx in range(num_time_points):
                # Get EEG signal data at this time point across all channels
                eeg_signals = trials[trial_idx, :, time_idx]

                # Convert to string representation (comma-separated values in brackets)
                eeg_signals_str = "[" + ",".join([f"{val:.6f}" for val in eeg_signals]) + "]"

                rows.append({
                    "motion_type": motion_type,
                    "trial_index": trial_idx,
                    "time_point": time_idx,
                    "eeg_signals": eeg_signals_str
                })

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)


download_service = DownloadService()
