
"""
Author: @De oliveira Léna & @Sron Sarah
Date: 04/06/2025
Data augmentation module for hand sign language recognition.

This module provides various data augmentation techniques to improve
model generalization and robustness.


"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random

from utils import get_logger


class DataAugmentor:
    """Data augmentation for hand sign language sequences."""
    
    def __init__(self, config_dict: Union[Dict, None] = None, logger: logging.Logger = None):
        """Initialize the data augmentor.
        
        Args:
            config_dict: Configuration dictionary containing augmentation parameters
            logger: Logger instance for recording operations
        """
        self.logger = logger or get_logger(__name__)
        
        if config_dict is None:
            self.logger.info("No config dictionary provided, using default values.")
            config_dict = {
                "enable_augmentation": True,
                "augmentation_probability": 0.5,
                "enable_noise": True,
                "noise_std": 0.01,
                "enable_time_warping": True,
                "time_warp_sigma": 0.2,
                "time_warp_knot": 4,
                "enable_magnitude_warping": True,
                "magnitude_warp_sigma": 0.2,
                "magnitude_warp_knot": 4,
                "enable_window_slicing": True,
                "slice_ratio": 0.1,
                "enable_spatial_transform": True,
                "rotation_range": 0.1,
                "scale_range": 0.1,
                "translation_range": 0.05,
                "enable_speed_variation": True,
                "speed_range": [0.8, 1.2]
            }
        else:
            self.logger.info(f"Using provided config dictionary: {config_dict}")
            
        # Initialize configuration parameters
        self.enable_augmentation = config_dict.get("enable_augmentation", True)
        self.augmentation_probability = config_dict.get("augmentation_probability", 0.5)
        self.enable_noise = config_dict.get("enable_noise", True)
        self.noise_std = config_dict.get("noise_std", 0.01)
        self.enable_time_warping = config_dict.get("enable_time_warping", True)
        self.time_warp_sigma = config_dict.get("time_warp_sigma", 0.2)
        self.time_warp_knot = config_dict.get("time_warp_knot", 4)
        self.enable_magnitude_warping = config_dict.get("enable_magnitude_warping", True)
        self.magnitude_warp_sigma = config_dict.get("magnitude_warp_sigma", 0.2)
        self.magnitude_warp_knot = config_dict.get("magnitude_warp_knot", 4)
        self.enable_window_slicing = config_dict.get("enable_window_slicing", True)
        self.slice_ratio = config_dict.get("slice_ratio", 0.1)
        self.enable_spatial_transform = config_dict.get("enable_spatial_transform", True)
        self.rotation_range = config_dict.get("rotation_range", 0.1)
        self.scale_range = config_dict.get("scale_range", 0.1)
        self.translation_range = config_dict.get("translation_range", 0.05)
        self.enable_speed_variation = config_dict.get("enable_speed_variation", True)
        self.speed_range = config_dict.get("speed_range", [0.8, 1.2])
        
        self.logger.info(f"DataAugmentor initialized with config: {config_dict}")
        
    def augment_sequence(self, sequence: np.ndarray, 
                        augmentation_types: Optional[List[str]] = None) -> np.ndarray:
        """Apply augmentation to a single sequence.
        
        Args:
            sequence: Input sequence with shape (sequence_length, feature_dim)
            augmentation_types: List of specific augmentations to apply
            
        Returns:
            Augmented sequence
        """
        # TODO: Implement sequence augmentation
        # 1. Check if augmentation is enabled and apply probability check
        if not self.enable_augmentation:
            return sequence

        if random.random() > self.augmentation_probability:
            return sequence

        # 2. Define available augmentation methods dictionary
        available_augmentations = {
            "noise": self._add_noise,
            "time_warping": self._time_warping,
            "magnitude_warping": self._magnitude_warping,
            "window_slicing": self._window_slicing,
            "spatial_transform": self._spatial_transformation,
            "speed_variation": self._speed_variation
        }

        # 3. Select augmentation methods (randomly or from specified list)
        if augmentation_types is None:
            augmentation_types = []
            if self.enable_noise:
                augmentation_types.append("noise")
            if self.enable_time_warping:
                augmentation_types.append("time_warping")
            if self.enable_magnitude_warping:
                augmentation_types.append("magnitude_warping")
            if self.enable_window_slicing:
                augmentation_types.append("window_slicing")
            if self.enable_spatial_transform:
                augmentation_types.append("spatial_transform")
            if self.enable_speed_variation:
                augmentation_types.append("speed_variation")

        random.shuffle(augmentation_types)

        # 4. Apply selected augmentations sequentially
        augmented_sequence = sequence.copy()
        for aug_type in augmentation_types:
            if aug_type in available_augmentations:
                augmented_sequence = available_augmentations[aug_type](augmented_sequence)

        # 5. Return augmented sequence
        return augmented_sequence
        
    def _add_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Noisy sequence
        """
        # TODO: Implement noise addition
        # 1. Generate Gaussian noise with specified std
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=sequence.shape)
        
        # 2. Only add noise to non-zero values
        mask = sequence != 0
        noisy_sequence = sequence.copy()
        noisy_sequence[mask] += noise[mask]
        
        # 3. Return noisy sequence
        return noisy_sequence

        
    def _time_warping(self, sequence: np.ndarray) -> np.ndarray:
        """Apply time warping to the sequence.

        Args:
            sequence: Input sequence

        Returns:
            Time-warped sequence
        """
        # TODO: Implement time warping
        # 1. Generate random warping factors
        orig_steps = np.arange(sequence.shape[0])
        random_warp = np.random.normal(loc=1.0, scale=self.time_warp_sigma, size=sequence.shape[0])
        warped_steps = np.cumsum(random_warp)
        warped_steps = np.interp(orig_steps, np.linspace(0, sequence.shape[0] - 1, num=len(warped_steps)), warped_steps)

        # 2. Create time points for interpolation
        warped_steps = (warped_steps - warped_steps.min()) / (warped_steps.max() - warped_steps.min()) * (sequence.shape[0] - 1)
        target_steps = np.arange(sequence.shape[0])

        # 3. Apply warping through interpolation
        warped_sequence = np.zeros_like(sequence)
        for d in range(sequence.shape[1]):
            warped_sequence[:, d] = np.interp(target_steps, warped_steps, sequence[:, d])

        # 4. Return warped sequence
        return warped_sequence

        
    def _magnitude_warping(self, sequence: np.ndarray) -> np.ndarray:
        """Apply magnitude warping to the sequence.

        Args:
            sequence: Input sequence

        Returns:
            Magnitude-warped sequence
        """
        # TODO: Implement magnitude warping
        # 1. Generate smooth random warping curve
        seq_len, feat_dim = sequence.shape
        time_steps = np.arange(seq_len)

        # On crée un facteur de déformation aléatoire lissé (via interpolation spline)
        random_factors = np.random.normal(loc=1.0, scale=self.magnitude_warp_sigma, size=(self.magnitude_warp_knot, feat_dim))
        knot_positions = np.linspace(0, seq_len - 1, self.magnitude_warp_knot)

        warping_curve = np.zeros_like(sequence)
        for i in range(feat_dim):
            warping_curve[:, i] = np.interp(time_steps, knot_positions, random_factors[:, i])

        # 2. Apply warping to non-zero values
        warped = sequence.copy()
        mask = sequence != 0
        warped[mask] *= warping_curve[mask]

        # 3. Return warped sequence
        return warped

        
    def _window_slicing(self, sequence: np.ndarray) -> np.ndarray:
        """Apply window slicing to the sequence.

        Args:
            sequence: Input sequence

        Returns:
            Sliced sequence
        """
        # TODO: Implement window slicing
        seq_len, feat_dim = sequence.shape

        # 1. Calculate slice length based on ratio
        slice_len = int(seq_len * self.slice_ratio)
        if slice_len == 0 or slice_len >= seq_len:
            return sequence.copy()

        # 2. Randomly choose slice position
        start_idx = np.random.randint(0, seq_len - slice_len)

        # 3. Remove selected window
        sliced = np.delete(sequence, np.s_[start_idx:start_idx + slice_len], axis=0)

        # 4. Pad or truncate to maintain original length
        if sliced.shape[0] < seq_len:
            pad_len = seq_len - sliced.shape[0]
            pad = np.zeros((pad_len, feat_dim))
            sliced = np.vstack([sliced, pad])

        # 5. Return sliced sequence
        return sliced

        
    def _spatial_transformation(self, sequence: np.ndarray) -> np.ndarray:
        """Apply spatial transformation to the sequence.

        Args:
            sequence: Input sequence

        Returns:
            Spatially transformed sequence
        """
        # TODO: Implement spatial transformation
        # 1. Generate random transformation parameters (rotation, scale, translation)
        rotation = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)

        # 2. Create transformation matrix
        transform = np.array([
            [scale * np.cos(rotation), -scale * np.sin(rotation)],
            [scale * np.sin(rotation),  scale * np.cos(rotation)]
        ])

        # 3. Apply transformation to coordinate features
        transformed_sequence = sequence.copy()
        feature_dim = transformed_sequence.shape[1]

        # 4. Handle different feature groups (pose, face, hands)
        # We assume 3D keypoints ordered as x1, y1, z1, x2, y2, z2, ..., so we reshape
        reshaped = transformed_sequence.reshape(transformed_sequence.shape[0], -1, 3)  # shape: (T, num_points, 3)
        
        for t in range(reshaped.shape[0]):
            coords = reshaped[t, :, :2]  # Only apply on (x, y)
            coords = np.dot(coords, transform.T)
            coords += np.array([tx, ty])
            reshaped[t, :, :2] = coords  # Replace only x, y

        # 5. Return transformed sequence
        return reshaped.reshape(sequence.shape)

        
    def _speed_variation(self, sequence: np.ndarray) -> np.ndarray:
        """Apply speed variation to the sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Speed-varied sequence
        """
        # TODO: Implement speed variation
        # 1. Generate random speed factor
        speed = np.random.uniform(self.speed_range[0], self.speed_range[1])
        
        # 2. Create new time indices based on speed
        original_length = sequence.shape[0]
        new_length = int(original_length / speed)
        new_length = max(2, new_length)  # éviter new_length=1

        # 3. Interpolate to new time points
        orig_indices = np.linspace(0, 1, original_length)
        new_indices = np.linspace(0, 1, new_length)
        interpolated = np.zeros((new_length, sequence.shape[1]))

        for i in range(sequence.shape[1]):
            interpolated[:, i] = np.interp(new_indices, orig_indices, sequence[:, i])

        # 4. Pad or truncate to maintain original length
        if new_length > original_length:
            interpolated = interpolated[:original_length]
        elif new_length < original_length:
            pad_len = original_length - new_length
            pad = np.zeros((pad_len, sequence.shape[1]))
            interpolated = np.concatenate([interpolated, pad], axis=0)

        # 5. Return speed-varied sequence
        return interpolated

        
    def augment_dataset(self, sequences: np.ndarray, labels: np.ndarray, 
                        augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Augment entire dataset.
        
        Args:
            sequences: Input sequences with shape (n_sequences, sequence_length, feature_dim)
            labels: Corresponding labels
            augmentation_factor: Number of augmented samples per original sample
            
        Returns:
            Tuple of (augmented_sequences, augmented_labels)
        """
        # TODO: Implement dataset augmentation
        # 1. Check if augmentation is enabled
        if not self.enable_augmentation or augmentation_factor <= 0:
            return sequences, labels

        self.logger.info(f"Augmenting dataset with factor {augmentation_factor}")

        # 2. For each augmentation round:
        #    - Apply augmentation to each sequence
        #    - Collect augmented sequences and labels
        augmented_seqs = []
        augmented_labs = []

        for i in range(augmentation_factor):
            self.logger.info(f"Augmentation round {i+1}/{augmentation_factor}")
            for seq, label in zip(sequences, labels):
                aug_seq = self.augment_sequence(seq)
                augmented_seqs.append(aug_seq)
                augmented_labs.append(label)

        # 3. Combine original and augmented data
        all_sequences = np.concatenate([sequences, np.array(augmented_seqs)], axis=0)
        all_labels = np.concatenate([labels, np.array(augmented_labs)], axis=0)

        # 4. Return final augmented dataset
        return all_sequences, all_labels


    
