
"""
Author: @De oliveira Léna & @Sron Sarah 
Date: 03/06/2025
Data preprocessing module for hand sign language recognition.

This module provides comprehensive data preprocessing capabilities including
data cleaning, normalization, validation, and quality assessment.

"""

import logging
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import json

from utils import get_logger


class DataPreprocessor:
    """Comprehensive data preprocessor for hand sign language data."""
    
    def __init__(self, config_dict: Union[Dict, None] = None, logger: logging.Logger = None):
        """Initialize the data preprocessor.
        
        Args:
            config_dict: Configuration dictionary containing preprocessing parameters
            logger: Logger instance for recording operations
        """
        self.logger = logger or get_logger(__name__)
        
        if config_dict is None:
            self.logger.info("No config dictionary provided, using default values.")
            config_dict = {
                "sequence_length": 30,
                "feature_dim": 1662,
                "min_confidence_threshold": 0.5,
                "max_zero_ratio": 0.3,
                "outlier_threshold": 3.0,
                "enable_normalization": True,
                "normalization_method": "standard",
                "enable_smoothing": True,
                "smoothing_window": 3,
                "enable_interpolation": True,
                "enable_quality_check": True,
                "save_quality_report": True
            }
        else:
            self.logger.info(f"Using provided config dictionary: {config_dict}")
            
        # Initialize configuration parameters
        self.sequence_length = config_dict.get("sequence_length", 30)
        self.feature_dim = config_dict.get("feature_dim", 1662)
        self.min_confidence_threshold = config_dict.get("min_confidence_threshold", 0.5)
        self.max_zero_ratio = config_dict.get("max_zero_ratio", 0.3)
        self.outlier_threshold = config_dict.get("outlier_threshold", 3.0)
        self.enable_normalization = config_dict.get("enable_normalization", True)
        self.normalization_method = config_dict.get("normalization_method", "standard")
        self.enable_smoothing = config_dict.get("enable_smoothing", True)
        self.smoothing_window = config_dict.get("smoothing_window", 3)
        self.enable_interpolation = config_dict.get("enable_interpolation", True)
        self.enable_quality_check = config_dict.get("enable_quality_check", True)
        self.save_quality_report = config_dict.get("save_quality_report", True)
        
        # Initialize scaler
        self._init_scalers()
        
        self.logger.info(f"DataPreprocessor initialized with config: {config_dict}")
        
    def _init_scalers(self) -> None:
        """Initialize normalization scalers."""
        if self.normalization_method == "standard":
            self.scaler = StandardScaler()
        elif self.normalization_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")
            
    def load_sequence_data(self, data_path: str, word: str) -> Tuple[np.ndarray, List[str]]:
        """Load all sequences for a specific word."""
        
        word_path = os.path.join(data_path, word)
        if not os.path.exists(word_path):
            self.logger.warning(f"Word directory not found: {word_path}")
            return np.empty((0, self.sequence_length, self.feature_dim)), []

        sequences_list = []
        sequence_info = []

        for seq_folder in sorted(os.listdir(word_path)):
            seq_path = os.path.join(word_path, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            frames = []
            for i in range(self.sequence_length):
                frame_path = os.path.join(seq_path, f"{i}.npy")
                if os.path.exists(frame_path):
                    try:
                        keypoints = np.load(frame_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {frame_path}: {e}")
                        keypoints = np.zeros(self.feature_dim)
                else:
                    keypoints = np.zeros(self.feature_dim)
                    self.logger.warning(f"Missing frame {i} in {seq_path}, filled with zeros")
                frames.append(keypoints)

            try:
                sequence_array = np.stack(frames)
                sequences_list.append(sequence_array)
                sequence_info.append(f"{word}_{seq_folder}")
            except Exception as e:
                self.logger.warning(f"Failed to stack frames in {seq_path}: {e}")
                continue

        if sequences_list:
            sequences = np.stack(sequences_list)
        else:
            sequences = np.empty((0, self.sequence_length, self.feature_dim))

        self.logger.info(f"Loaded {len(sequences)} sequences for word: {word}")
        self.logger.info(f"Shape of loaded data: {sequences.shape}")

        return sequences, sequence_info
        
    def validate_sequence(self, sequence: np.ndarray, sequence_id: str = "") -> Dict:
        """Validate a single sequence for quality issues.
        
        Args:
            sequence: Input sequence with shape (sequence_length, feature_dim)
            sequence_id: Identifier for the sequence
            
        Returns:
            Dictionary containing validation results
        """
        # 1. Check sequence shape
        if sequence.shape != (self.sequence_length, self.feature_dim):
            return {
                "sequence_id": sequence_id,
                "is_valid": False,
                "issues": ["Invalid shape"],
                "quality_score": 0.0,
                "statistics": {}
            }

        validation_result = {
            "sequence_id": sequence_id,
            "is_valid": True,
            "issues": [],
            "quality_score": 1.0,
            "statistics": {}
        }

        # 2. Calculate zero ratio
        total_values = np.prod(sequence.shape)
        zero_count = np.sum(sequence == 0)
        zero_ratio = zero_count / total_values
        validation_result["statistics"]["zero_ratio"] = zero_ratio

        if zero_ratio > self.max_zero_ratio:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Too many zeros")

        # 3. Detect outliers using z-score
        flattened = sequence[sequence != 0]  # ignore zeros
        if flattened.size > 0:
            z_scores = np.abs((flattened - np.mean(flattened)) / np.std(flattened))
            outlier_ratio = np.mean(z_scores > 3)
        else:
            outlier_ratio = 1.0

        validation_result["statistics"]["outlier_ratio"] = outlier_ratio

        if outlier_ratio > 0.1:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Too many outliers")

        # 4. Check for missing frames (all zeros)
        missing_frames = np.sum(np.all(sequence == 0, axis=1))
        validation_result["statistics"]["missing_frames"] = int(missing_frames)

        if missing_frames > 0:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Missing frames")

        # 5. Calculate a simple quality score (example heuristic)
        score = 1.0 - (zero_ratio + outlier_ratio + 0.05 * missing_frames)
        validation_result["quality_score"] = max(0.0, min(1.0, score))

        self.logger.debug(f"Validation result for {sequence_id}: {validation_result}")

        return validation_result

        
    def _detect_outliers(self, sequence: np.ndarray) -> float:
        """Detect outliers using z-score method.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Ratio of outlier values
        """
        # TODO: Implement outlier detection
        # 1. Get non-zero values from sequence
        # 2. Calculate z-scores
        # 3. Count outliers above threshold
        # 4. Return outlier ratio
        
        return 0.0
        
    def clean_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Clean a single sequence by handling missing values and outliers.
    
        Args:
            sequence: Input sequence with shape (sequence_length, feature_dim)
        
        Returns:
            Cleaned sequence
        
        """
    
        # TODO: Implement sequence cleaning
        # 1. Handle missing values with interpolation (if enabled)
        # 2. Apply smoothing filter (if enabled)
        # 3. Return cleaned sequence
        
        cleaned_sequence = sequence.copy()
        
        # Étape 1 : interpolation des valeurs manquantes
        if self.enable_interpolation:
            cleaned_sequence = self._interpolate_missing_values(cleaned_sequence)
        
        # Étape 2 : filtre de lissage
        if self.enable_smoothing:
            cleaned_sequence = self._smooth_sequence(cleaned_sequence)
        
        return cleaned_sequence

        
    def _interpolate_missing_values(self, sequence: np.ndarray) -> np.ndarray:
        """Interpolate missing values (zeros and NaNs) in the sequence.

        Args:
            sequence: Input sequence
            
        Returns:
            Interpolated sequence
        """
        # 1. Copy the sequence to avoid in-place modification
        interpolated = sequence.copy()

        # 2. Loop over all features (i.e., columns)
        for i in range(interpolated.shape[1]):
            feature = interpolated[:, i]

            # 3. Identify valid (non-zero and non-NaN) values
            valid_idx = np.where((feature != 0) & (~np.isnan(feature)))[0]

            # 4. Interpolate only if we have at least two valid points
            if len(valid_idx) > 1:
                interpolated[:, i] = np.interp(
                    np.arange(len(feature)),
                    valid_idx,
                    feature[valid_idx]
                )
            elif len(valid_idx) == 1:
                # 5. If only one valid point, fill entire column with that value
                interpolated[:, i] = feature[valid_idx[0]]
            else:
                # 6. If no valid point, fill with zeros
                interpolated[:, i] = 0.0

        return interpolated

        
    def _smooth_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply smoothing filter to reduce noise.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Smoothed sequence
        """
        # TODO: Implement sequence smoothing
        # 1. Apply median filter for each feature using scipy.signal.medfilt
        # 2. Only smooth non-zero values
        # 3. Return smoothed sequence

        smoothed = sequence.copy()

        for i in range(smoothed.shape[1]):  # pour chaque colonne (feature)
            feature = smoothed[:, i]

            # Créer un masque des valeurs non nulles
            nonzero_mask = feature != 0

            # Si au moins 3 valeurs non nulles pour appliquer un filtre de taille 3
            if np.sum(nonzero_mask) >= 3:
                # Appliquer un filtre médian sur l'ensemble
                filtered = signal.medfilt(feature, kernel_size=self.smoothing_window)
                # Remplacer uniquement les valeurs non nulles par les valeurs filtrées
                feature[nonzero_mask] = filtered[nonzero_mask]
                smoothed[:, i] = feature

        return smoothed

    def normalize_sequences(self, sequences: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Normalize sequences using the configured method."""
        if not self.enable_normalization:
            return sequences

        self.logger.info(f"Normalizing {len(sequences)} sequences using {self.normalization_method} method")

        # 1. Reshape to (n_samples, feature_dim)
        n_seq, seq_len, feat_dim = sequences.shape
        flat = sequences.reshape(-1, feat_dim)

        # 2. Supprimer les lignes qui sont toutes à zéro
        non_zero_rows = ~np.all(flat == 0, axis=1)
        data_to_scale = flat[non_zero_rows]

        # 3. Fit le scaler si demandé
        if fit_scaler:
            self._init_scalers()
            self.scaler.fit(data_to_scale)

        # 4. Appliquer le scaler uniquement sur les lignes non nulles
        flat_scaled = flat.copy()
        flat_scaled[non_zero_rows] = self.scaler.transform(data_to_scale)

        # 5. Reformer la forme d'origine
        return flat_scaled.reshape(n_seq, seq_len, feat_dim)

        
    def process_dataset(self, data_path: str, words: List[str], 
                    output_path: Optional[str] = None) -> Dict:
        """Process complete dataset with all preprocessing steps."""
        
        self.logger.info(f"Processing dataset for {len(words)} words")

        # 1. Initialize result dictionary
        processed_data = {
            "sequences": [],
            "labels": [],
            "word_to_idx": {},
            "statistics": {},
            "quality_report": []
        }

        # 2. Create word to index mapping
        word_to_idx = {word: idx for idx, word in enumerate(words)}
        processed_data["word_to_idx"] = word_to_idx

        # 3. For each word:
        for word in words:
            self.logger.info(f"Processing word: {word}")
            sequences, infos = self.load_sequence_data(data_path, word)

            valid_sequences = []
            labels = []

            for i, sequence in enumerate(sequences):
                seq_id = infos[i] if i < len(infos) else f"{word}_{i}"
                result = self.validate_sequence(sequence, seq_id)
                processed_data["quality_report"].append(result)

                if result["is_valid"]:
                    cleaned = self.clean_sequence(sequence)
                    valid_sequences.append(cleaned)
                    labels.append(word_to_idx[word])
            
            if valid_sequences:
                processed_data["sequences"].extend(valid_sequences)
                processed_data["labels"].extend(labels)
                processed_data["statistics"][word] = len(valid_sequences)

        # 4. Normalize all sequences
        if processed_data["sequences"]:
            sequences_array = np.stack(processed_data["sequences"])
            normalized = self.normalize_sequences(sequences_array)
            processed_data["sequences"] = normalized
            processed_data["labels"] = np.array(processed_data["labels"])
        else:
            processed_data["sequences"] = np.empty((0, self.sequence_length, self.feature_dim))
            processed_data["labels"] = np.array([])

        # 5. Save processed data if output_path provided
        if output_path:
            self._save_processed_data(processed_data, output_path)

        # 6. Return processed data dictionary
        return processed_data

        
    def _save_processed_data(self, processed_data: Dict, output_path: str) -> None:
        print(">>> Fonction _save_processed_data appelée")

        """Save processed data to disk."""
        self.logger.info(f"Saving processed data to: {output_path}")

        os.makedirs(output_path, exist_ok=True)

        # 1. Save sequences and labels
        np.save(os.path.join(output_path, "sequences.npy"), processed_data["sequences"])
        np.save(os.path.join(output_path, "labels.npy"), processed_data["labels"])

        # 2. Save metadata (word_to_idx, statistics)
        metadata = {
            "word_to_idx": processed_data["word_to_idx"],
            "statistics": processed_data["statistics"]
        }
        with open(os.path.join(output_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 3. Optionally save quality report
        if "quality_report" in processed_data:
            with open(os.path.join(output_path, "quality_report.json"), "w") as f:
                json.dump(processed_data["quality_report"], f, indent=2)
