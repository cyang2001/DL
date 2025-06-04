"""
Author: @De oliveira Léna & @Sron Sarah
Date: 05/06/2025
Preprocessing pipeline for hand sign language recognition.

This module provides a complete preprocessing pipeline that integrates
data cleaning, augmentation, and feature engineering.

"""

import logging
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split

from utils import get_logger
from .data_preprocessor import DataPreprocessor
from .data_augmentor import DataAugmentor
from .feature_engineer import FeatureEngineer


class PreprocessingPipeline:
    """Complete preprocessing pipeline for hand sign language data."""
    
    def __init__(self, config_dict: Union[Dict, None] = None, logger: logging.Logger = None):
        """Initialize the preprocessing pipeline.
        
        Args:
            config_dict: Configuration dictionary containing all preprocessing parameters
            logger: Logger instance for recording operations
        """
        self.logger = logger or get_logger(__name__)
        
        if config_dict is None:
            self.logger.info("No config dictionary provided, using default values.")
            config_dict = {
                "raw_data_path": "MP_data",
                "processed_data_path": "processed_data",
                "validation_split": 0.2,
                "test_split": 0.1,
                "random_seed": 42,
                "preprocessing": {
                    "enable_normalization": True,
                    "normalization_method": "standard",
                    "enable_smoothing": True,
                    "enable_interpolation": True
                },
                "augmentation": {
                    "enable_augmentation": True,
                    "augmentation_factor": 2
                },
                "feature_engineering": {
                    "enable_feature_engineering": False
                }
            }
        else:
            self.logger.info(f"Using provided config dictionary: {config_dict}")
        
        self.config_dict = config_dict
        
        self.data_preprocessor = DataPreprocessor(
            config_dict.get("preprocessing", {}), self.logger)
        self.data_augmentor = DataAugmentor(
            config_dict.get("augmentation", {}), self.logger)
        self.feature_engineer = FeatureEngineer(
            config_dict.get("feature_engineering", {}), self.logger)
        
        # Resolve paths relative to project root
        self.raw_data_path = self._resolve_path(config_dict.get("raw_data_path", "MP_data"))
        self.processed_data_path = self._resolve_path(config_dict.get("processed_data_path", "processed_data"))
        self.validation_split = config_dict.get("validation_split", 0.2)
        self.test_split = config_dict.get("test_split", 0.1)
        self.random_seed = config_dict.get("random_seed", 42)
        
        self.logger.info("PreprocessingPipeline initialized successfully")
        self.logger.info(f"Raw data path: {self.raw_data_path}")
        self.logger.info(f"Processed data path: {self.processed_data_path}")
        
    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to project root directory.
        
        Args:
            path: Input path (can be relative or absolute)
            
        Returns:
            Resolved absolute path
        """
        if os.path.isabs(path):
            return path
            
        # Find project root (directory containing utils.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        
        # Navigate up to find project root
        while project_root != "/" and not os.path.exists(os.path.join(project_root, "utils.py")):
            project_root = os.path.dirname(project_root)
            
        if not os.path.exists(os.path.join(project_root, "utils.py")):
            self.logger.warning("Could not find project root (utils.py). Using current working directory.")
            project_root = os.getcwd()
            
        resolved_path = os.path.join(project_root, path)
        return os.path.abspath(resolved_path)
        
    def process_full_pipeline(self, words: List[str], save_processed: bool = True) -> Dict:
        """Execute the complete preprocessing pipeline.

        Args:
            words: List of words to process
            save_processed: Whether to save processed data

        Returns:
            Dictionary containing processed dataset splits
        """
        self.logger.info(f"Starting full preprocessing pipeline for {len(words)} words")

        # 1. Load and clean raw data using data_preprocessor
        processed = self.data_preprocessor.process_dataset(
            data_path=self.raw_data_path,
            words=words,
            output_path=self.processed_data_path
        )

        sequences = np.array(processed["sequences"])
        labels_str = processed["labels"]
        word_to_idx = processed["word_to_idx"]
        quality_report = processed["quality_report"]

        labels = np.array(labels_str)


        # 2. Apply feature engineering if enabled
        if self.feature_engineer.enable_feature_engineering:
            sequences = np.array([self.feature_engineer.extract_features(seq) for seq in sequences])

        # 3. Split data into train/val/test
        splits = self._split_dataset(sequences, labels)

        # 4. Apply data augmentation to training data
        if self.data_augmentor.enable_augmentation:
            augmented_X, augmented_y = self.data_augmentor.augment_data(splits["X_train"], splits["y_train"])
            splits["X_train"] = np.concatenate([splits["X_train"], augmented_X], axis=0)
            splits["y_train"] = np.concatenate([splits["y_train"], augmented_y], axis=0)

        # 5. Compute final statistics
        stats = self._compute_final_statistics(splits, processed)

        # 6. Save processed data if requested
        result = {
            "X_train": splits["X_train"],
            "y_train": splits["y_train"],
            "X_val": splits["X_val"],
            "y_val": splits["y_val"],
            "X_test": splits["X_test"],
            "y_test": splits["y_test"],
            "word_to_idx": word_to_idx,
            "statistics": stats,
            "quality_report": quality_report
        }

        if save_processed:
            self._save_processed_dataset(result)

        # 7. Return result dictionary
        self.logger.info("Preprocessing pipeline completed successfully")
        return result

    
    def _split_dataset(self, sequences: np.ndarray, labels: np.ndarray) -> Dict:
        """Split dataset into train/validation/test sets.

        Args:
            sequences: All sequences
            labels: All labels

        Returns:
            Dictionary containing data splits
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        # 1. Encode string labels
        label_encoder = LabelEncoder()
        integer_labels = label_encoder.fit_transform(labels)  # ex: ['a', 'b'] -> [0, 1]

        # 2. One-hot encoding
        one_hot_encoder = OneHotEncoder(sparse_output=False)

        one_hot_labels = one_hot_encoder.fit_transform(integer_labels.reshape(-1, 1))

        # 3. Split test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, one_hot_labels,
            test_size=self.test_split,
            stratify=integer_labels,
            random_state=self.random_seed
        )

        # 4. Split train/val
        val_ratio = self.validation_split / (1.0 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=self.random_seed
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }
    
    def _compute_final_statistics(self, data_splits: Dict, processed_data: Dict) -> Dict:
        """Compute final dataset statistics.

        Args:
            data_splits: Data split dictionary
            processed_data: Original processed data

        Returns:
            Final statistics dictionary
        """
        # 1. Calculate dataset sizes
        total_sequences = (
            data_splits["X_train"].shape[0] +
            data_splits["X_val"].shape[0] +
            data_splits["X_test"].shape[0]
        )

        train_size = data_splits["X_train"].shape[0]
        val_size = data_splits["X_val"].shape[0]
        test_size = data_splits["X_test"].shape[0]

        # 2. Compute feature dimensions
        sequence_length = data_splits["X_train"].shape[1]
        feature_dim = data_splits["X_train"].shape[2]

        # 3. Calculate class distribution
        train_class_dist = np.sum(data_splits["y_train"], axis=0).tolist()
        val_class_dist = np.sum(data_splits["y_val"], axis=0).tolist()
        test_class_dist = np.sum(data_splits["y_test"], axis=0).tolist()

        class_distribution = {
            "train": train_class_dist,
            "val": val_class_dist,
            "test": test_class_dist
        }

        # 4. Compute data quality metrics (fictif ici)
        quality_scores = processed_data.get("quality_scores", [])
        if quality_scores:
            avg_quality = float(np.mean(quality_scores))
            valid_ratio = float(np.mean([score >= 0.5 for score in quality_scores]))
        else:
            avg_quality = 0.0
            valid_ratio = 1.0

        # 5. Return statistics dictionary
        stats = {
            "dataset_size": {
                "total_sequences": total_sequences,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size
            },
            "feature_dimensions": {
                "sequence_length": sequence_length,
                "feature_dim": feature_dim
            },
            "class_distribution": class_distribution,
            "data_quality": {
                "average_quality_score": avg_quality,
                "valid_sequences_ratio": valid_ratio
            }
        }

        return stats
   
    def _save_processed_dataset(self, result: Dict) -> None:
        """Save processed dataset to disk.

        Args:
            result: Complete processing result
        """
        # TODO: Implement data saving
        # 1. Create output directory
        os.makedirs(self.processed_data_path, exist_ok=True)

        # 2. Save data splits as .npy files
        for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
            np.save(os.path.join(self.processed_data_path, f"{key}.npy"), result[key])

        # 3. Save metadata as JSON
        import json
        metadata = {
            "word_to_idx": result.get("word_to_idx", {}),
            "statistics": result.get("statistics", {})
        }
        with open(os.path.join(self.processed_data_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # 4. Save quality report if available
        quality = result.get("quality_report", [])
        if quality:
            with open(os.path.join(self.processed_data_path, "quality_report.json"), "w") as f:
                json.dump(quality, f, indent=4)

        self.logger.info(f"Processed dataset saved to: {self.processed_data_path}")
 
    def load_processed_dataset(self) -> Optional[Dict]:
        """Load previously processed dataset.

        Returns:
            Loaded dataset or None if not found
        """
        if not os.path.exists(self.processed_data_path):
            self.logger.warning(f"Processed data directory not found: {self.processed_data_path}")
            return None

        try:
            # Charger les fichiers .npy
            data_splits = {}
            for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
                file_path = os.path.join(self.processed_data_path, f"{key}.npy")
                data_splits[key] = np.load(file_path)

            # Charger les métadonnées
            with open(os.path.join(self.processed_data_path, "metadata.json"), "r") as f:
                metadata = json.load(f)

            # Charger le rapport qualité s'il existe
            quality_path = os.path.join(self.processed_data_path, "quality_report.json")
            if os.path.exists(quality_path):
                with open(quality_path, "r") as f:
                    quality_report = json.load(f)
            else:
                quality_report = []

            # Regrouper tous les résultats
            result = {
                **data_splits,
                "word_to_idx": metadata.get("word_to_idx", {}),
                "statistics": metadata.get("statistics", {}),
                "quality_report": quality_report
            }

            self.logger.info(f"✅ Dataset chargé depuis : {self.processed_data_path}")
            return result

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du dataset : {e}")
            return None
        
    def preprocess_single_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Preprocess a single sequence for real-time inference.

        Args:
            sequence: Raw sequence with shape (sequence_length, feature_dim)

        Returns:
            Preprocessed sequence
        """
        # 1. Nettoyage (interpolation, lissage, etc.)
        cleaned = self.data_preprocessor.clean_sequence(sequence)

        # 2. Normalisation (utilisation d’un scaler déjà entraîné dans le préprocesseur)
        normalized = self.data_preprocessor.normalize_sequence(cleaned)

        # 3. Feature engineering (si activé dans la config)
        if self.feature_engineer.enable_feature_engineering:
            processed = self.feature_engineer.extract_features(normalized)
        else:
            processed = normalized

        # 4. Retourner la séquence prête pour l’inférence
        return processed
