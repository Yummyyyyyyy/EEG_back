"""
Augmentation service responsible for loading pre-generated samples
and aggregating them for visualization.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.models.augmentation import AugmentedChannels, AugmentedData, AugmentationRequest


class AugmentationService:
    """
    Service that loads locally generated augmentation samples
    and returns averaged channel curves for the requested method(s).
    """

    AUGMENTATION_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz"]
    STD_THRESHOLD = 1e-6

    MOTION_FILE_MAP = {
        "left": "left_1000.npz",
        "right": "right_1000.npz",
        "foot": "feet_1000.npz",
        "feet": "feet_1000.npz",
        "tongue": "tongue_1000.npz",
    }

    MOTION_EVENT_MAP = {
        "left": ("769", 769, 0),
        "right": ("770", 770, 1),
        "foot": ("771", 771, 2),
        "feet": ("771", 771, 2),
        "tongue": ("772", 772, 3),
    }

    DIFFUSION_FILE_MAP = {
        "left": "samples_class0_left.npz",
        "right": "samples_class1_right.npz",
        "foot": "samples_class2_feet.npz",
        "feet": "samples_class2_feet.npz",
        "tongue": "samples_class3_tongue.npz",
    }

    def __init__(self) -> None:
        base_dir = Path(__file__).parent.parent.parent
        self._tcn_dir = base_dir / "app/data/augmentation/tcn"
        self._gan_file = base_dir / "app/data/augmentation/gan/generated_trials.npz"
        self._vae_file = base_dir / "app/data/augmentation/vae/baseline_synthetic_generated_batch.npz"
        self._diffusion_dir = base_dir / "app/data/augmentation/diffusion"

        self._tcn_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._gan_cache: Optional[Dict[str, np.ndarray]] = None
        self._vae_cache: Optional[Dict[str, np.ndarray]] = None
        self._diffusion_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._channel_indices = self._resolve_channel_indices()

    def _resolve_channel_indices(self) -> Dict[str, int]:
        indices = {}
        for channel in self.AUGMENTATION_CHANNELS:
            if channel not in settings.CHANNEL_INDICES:
                raise KeyError(f"Missing channel index for {channel} in settings.CHANNEL_INDICES")
            indices[channel] = settings.CHANNEL_INDICES[channel]
        return indices

    def generate(self, request: AugmentationRequest) -> Dict[str, AugmentedData]:
        """
        Main entry point for augmentation generation.
        Currently only supports the TCN method.
        """
        result: Dict[str, AugmentedData] = {}

        for method in request.methods:
            normalized_method = method.lower()

            if normalized_method == "tcn":
                augmented = self._generate_tcn(request)
                result["tcn"] = augmented
            elif normalized_method == "gan":
                augmented = self._generate_gan(request)
                result["gan"] = augmented
            elif normalized_method == "vae":
                augmented = self._generate_vae(request)
                result["vae"] = augmented
            elif normalized_method == "diffusion":
                augmented = self._generate_diffusion(request)
                result["diffusion"] = augmented
            else:
                continue

        return result

    def _generate_tcn(self, request: AugmentationRequest) -> AugmentedData:
        """
        Aggregate pre-generated TCN samples for the requested motion type.
        """
        motion_type = request.eegData.trialInfo.motionType.lower()

        if motion_type not in self.MOTION_FILE_MAP:
            raise ValueError(f"Unsupported motion type for TCN augmentation: {motion_type}")

        samples_dict = self._load_tcn_samples(motion_type)
        eeg_samples = samples_dict["eeg_data"]
        labels = samples_dict.get("labels")

        filtered_samples = self._filter_by_motion_label(
            eeg_samples,
            labels,
            motion_type
        )

        if filtered_samples.shape[0] == 0:
            raise ValueError(f"No augmentation samples found for motion type: {motion_type}")

        count = min(request.count or 10, filtered_samples.shape[0])

        indices = random.sample(range(filtered_samples.shape[0]), count)
        selected = filtered_samples[indices, :, :]

        averaged_channels = self._average_channels(
            selected,
            request.eegData.channels,
            scale_to_microvolts=True,
            target_length=len(request.eegData.labels)
        )

        return AugmentedData(
            method="tcn",
            channels=AugmentedChannels(**averaged_channels)
        )

    def _generate_vae(self, request: AugmentationRequest) -> AugmentedData:
        """
        Aggregate VAE-generated samples for the requested motion type.
        """
        motion_type = request.eegData.trialInfo.motionType.lower()

        samples_dict = self._load_vae_samples()
        eeg_samples = samples_dict['trials']
        labels = samples_dict.get('labels')

        filtered_samples = self._filter_by_motion_label(
            eeg_samples,
            labels,
            motion_type
        )

        if filtered_samples.shape[0] == 0:
            raise ValueError(f"No VAE samples found for motion type: {motion_type}")

        count = min(request.count or 10, filtered_samples.shape[0])
        indices = random.sample(range(filtered_samples.shape[0]), count)
        selected = filtered_samples[indices, :, :, :] if filtered_samples.ndim == 4 else filtered_samples[indices, :, :]

        if selected.ndim == 4 and selected.shape[-1] == 1:
            selected = selected[..., 0]

        averaged_channels = self._average_channels(
            selected,
            request.eegData.channels,
            scale_to_microvolts=True,
            target_length=len(request.eegData.labels)
        )

        return AugmentedData(
            method='vae',
            channels=AugmentedChannels(**averaged_channels)
        )

    def _generate_diffusion(self, request: AugmentationRequest) -> AugmentedData:
        """
        Aggregate diffusion-model samples for the requested motion type.
        """
        motion_type = request.eegData.trialInfo.motionType.lower()

        samples_dict = self._load_diffusion_samples(motion_type)
        eeg_samples = samples_dict['trials']
        labels = samples_dict.get('labels')

        filtered_samples = self._filter_by_motion_label(
            eeg_samples,
            labels,
            motion_type
        )

        if filtered_samples.shape[0] == 0:
            raise ValueError(f"No diffusion samples found for motion type: {motion_type}")

        count = min(request.count or 10, filtered_samples.shape[0])
        indices = random.sample(range(filtered_samples.shape[0]), count)
        selected = filtered_samples[indices, :, :]

        averaged_channels = self._average_channels(
            selected,
            request.eegData.channels,
            scale_to_microvolts=True,
            target_length=len(request.eegData.labels)
        )

        return AugmentedData(
            method='diffusion',
            channels=AugmentedChannels(**averaged_channels)
        )

    def _generate_gan(self, request: AugmentationRequest) -> AugmentedData:
        """
        Aggregate GAN-generated samples for the requested motion type.
        """
        motion_type = request.eegData.trialInfo.motionType.lower()

        samples_dict = self._load_gan_samples()
        eeg_samples = samples_dict["trials"]
        labels = samples_dict.get("labels")

        filtered_samples = self._filter_by_motion_label(
            eeg_samples,
            labels,
            motion_type
        )

        if filtered_samples.shape[0] == 0:
            raise ValueError(f"No GAN samples found for motion type: {motion_type}")

        count = min(request.count or 10, filtered_samples.shape[0])
        indices = random.sample(range(filtered_samples.shape[0]), count)
        selected = filtered_samples[indices, :, :]

        averaged_channels = self._average_channels(
            selected,
            request.eegData.channels,
            scale_to_microvolts=False,
            target_length=len(request.eegData.labels)
        )

        return AugmentedData(
            method="gan",
            channels=AugmentedChannels(**averaged_channels)
        )

    def _load_tcn_samples(self, motion_type: str) -> Dict[str, np.ndarray]:
        """
        Load (and cache) pre-generated TCN samples for the given motion type.
        """
        cache_key = motion_type
        if cache_key in self._tcn_cache:
            return self._tcn_cache[cache_key]

        file_name = self.MOTION_FILE_MAP[motion_type]
        file_path = self._tcn_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"TCN augmentation data not found: {file_path}")

        with np.load(file_path) as data:
            if "eeg_data" not in data:
                raise KeyError(f"'eeg_data' not found in {file_path}")

            eeg_data = data["eeg_data"]
            labels = data.get("labels")

        cache_entry = {"eeg_data": eeg_data}
        if labels is not None:
            cache_entry["labels"] = labels

        self._tcn_cache[cache_key] = cache_entry
        return cache_entry

    def _load_vae_samples(self) -> Dict[str, np.ndarray]:
        """
        Load (and cache) VAE-generated samples.
        """
        if self._vae_cache is not None:
            return self._vae_cache

        if not self._vae_file.exists():
            raise FileNotFoundError(f"VAE augmentation data not found: {self._vae_file}")

        with np.load(self._vae_file) as data:
            if 'X_gen' not in data or 'y_gen' not in data:
                raise KeyError(f"'X_gen' or 'y_gen' not found in {self._vae_file}")

            trials = data['X_gen']
            labels = data['y_gen']

        self._vae_cache = {
            'trials': trials,
            'labels': labels
        }
        return self._vae_cache

    def _load_diffusion_samples(self, motion_type: str) -> Dict[str, np.ndarray]:
        """
        Load (and cache) diffusion model samples for the given motion type.
        """
        cache_key = motion_type
        if cache_key in self._diffusion_cache:
            return self._diffusion_cache[cache_key]

        file_name = self.DIFFUSION_FILE_MAP.get(motion_type)
        if not file_name:
            raise ValueError(f"Unsupported motion type for diffusion augmentation: {motion_type}")

        file_path = self._diffusion_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Diffusion augmentation data not found: {file_path}")

        with np.load(file_path) as data:
            if 'X' not in data:
                raise KeyError(f"'X' not found in {file_path}")
            trials = data['X']
            labels = data.get('y')

        cache_entry = {
            'trials': trials,
        }
        if labels is not None:
            cache_entry['labels'] = labels

        self._diffusion_cache[cache_key] = cache_entry
        return cache_entry

    def _load_gan_samples(self) -> Dict[str, np.ndarray]:
        """
        Load (and cache) GAN-generated samples.
        """
        if self._gan_cache is not None:
            return self._gan_cache

        if not self._gan_file.exists():
            raise FileNotFoundError(f"GAN augmentation data not found: {self._gan_file}")

        with np.load(self._gan_file) as data:
            if "trials" not in data or "labels" not in data:
                raise KeyError(f"'trials' or 'labels' not found in {self._gan_file}")

            trials = data["trials"]
            labels = data["labels"]

        self._gan_cache = {
            "trials": trials,
            "labels": labels
        }
        return self._gan_cache

    def _filter_by_motion_label(
        self,
        eeg_samples: np.ndarray,
        labels: np.ndarray | None,
        motion_type: str
    ) -> np.ndarray:
        """
        Filter EEG samples by MI label; fall back to all samples if labels missing.
        """
        if labels is None:
            return eeg_samples

        label_info = self.MOTION_EVENT_MAP.get(motion_type)
        if not label_info:
            return eeg_samples

        str_label, int_label, small_label = label_info

        if labels.dtype.kind in {"U", "S", "O"}:
            mask = labels == str_label
        elif labels.dtype.kind in {"i", "u"} and np.isin(small_label, labels):
            mask = labels == small_label
        else:
            mask = labels == int_label

        filtered = eeg_samples[mask]
        if filtered.shape[0] == 0:
            return eeg_samples

        return filtered

    def _average_channels(
        self,
        samples: np.ndarray,
        base_channels,
        *,
        scale_to_microvolts: bool = True,
        target_length: Optional[int] = None
    ) -> Dict[str, List[Optional[float]]]:
        """
        Compute the averaged curve for the configured channels.
        """
        averaged: Dict[str, List[Optional[float]]] = {}

        for channel, idx in self._channel_indices.items():
            channel_samples = samples[:, idx, :]
            mean_curve = channel_samples.mean(axis=0)
            std_curve = channel_samples.std(axis=0)

            if scale_to_microvolts:
                mean_curve = mean_curve * 1e6
                std_curve = std_curve * 1e6

            base_channel_values = getattr(base_channels, channel, None)
            target_len = target_length or (
                len(base_channel_values) if base_channel_values is not None else mean_curve.shape[0]
            )

            if target_len > 0 and mean_curve.shape[0] != target_len:
                old_axis = np.linspace(0.0, 1.0, mean_curve.shape[0])
                new_axis = np.linspace(0.0, 1.0, target_len)
                mean_curve = np.interp(new_axis, old_axis, mean_curve)
                std_curve = np.interp(new_axis, old_axis, std_curve)

            if base_channel_values is not None:
                base_array = np.asarray(base_channel_values, dtype=float)
                aug_std = float(np.std(mean_curve))
                base_std = float(np.std(base_array))
                if base_std > 0 and aug_std > 0:
                    scale = base_std / aug_std
                    mean_curve = mean_curve * scale
                    std_curve = std_curve * scale

            channel_list: List[Optional[float]] = []
            for value, std in zip(mean_curve, std_curve):
                if std < self.STD_THRESHOLD:
                    channel_list.append(None)
                else:
                    channel_list.append(float(value))

            if len(channel_list) < target_len:
                channel_list.extend([None] * (target_len - len(channel_list)))
            elif len(channel_list) > target_len:
                channel_list = channel_list[:target_len]

            averaged[channel] = channel_list

        return averaged


augmentation_service = AugmentationService()
