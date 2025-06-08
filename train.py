"""
Author: @Chen YANG and @ChatGPT
Date: 2025-06-08
Training script for hand sign language recognition.

This script loads processed data and trains the LSTM classifier.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from utils import get_logger
from src.preprocessing import PreprocessingPipeline
from src.classification.classifier import AttentionLSTMClassifier

logger = get_logger(__name__)

class TrainingManager:
    """Manager for training hand sign language recognition model."""
    
    def __init__(self, config: Dict = None):
        """Initialize training manager.
        
        Args:
            config: Training configuration dictionary
        """
        self.logger = logger
        
        if config is None:
            config = self._get_default_config()
            
        self.config = config
        self.preprocessing_pipeline = None
        self.classifier = None
        
        self.logger.info("TrainingManager initialized")
        
    def _get_default_config(self) -> Dict:
        """Get default training configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "data": {
                "raw_data_path": "MP_data",
                "processed_data_path": "processed_data",
                "validation_split": 0.2,
                "test_split": 0.1,
                "random_seed": 42
            },
            "preprocessing": {
                "sequence_length": 30,
                "feature_dim": 1662,
                "enable_normalization": True,
                "normalization_method": "standard",
                "enable_smoothing": True,
                "smoothing_window": 3,
                "enable_interpolation": True,
                "enable_quality_check": True,
                "max_zero_ratio": 0.3
            },
            "augmentation": {
                "enable_augmentation": True,
                "augmentation_factor": 2,
                "augmentation_probability": 0.5,
                "enable_noise": True,
                "noise_std": 0.01,
                "enable_time_warping": True,
                "enable_spatial_transform": True
            },
            "feature_engineering": {
                "enable_feature_engineering": False,
                "extract_velocity": True,
                "extract_acceleration": True,
                "extract_angles": True,
                "extract_distances": True
            },
            "model": {
                "lstm_units_1": 64,
                "lstm_units_2": 48,
                "dense_units": 32,
                "dropout": 0.2,
                "l2_regularization": 0.01,
                "learning_rate": 0.0005,
                "confidence_threshold": 0.7,
                "smoothing_window": 5
            },
            "training": {
                "epochs": 200,
                "batch_size": 16,
                "model_save_path": "models/best_hand_sign_model.h5",
                "plot_history": True
            }
        }
    
    def prepare_data(self, force_reprocess: bool = False) -> Dict:
        """Prepare training data using preprocessing pipeline.
        
        Args:
            force_reprocess: Whether to force reprocessing even if cached data exists
            
        Returns:
            Processed dataset dictionary
        """
        self.logger.info("Preparing training data...")
        
        # Initialize preprocessing pipeline
        preprocessing_config = {
            **self.config["data"],
            "preprocessing": self.config["preprocessing"],
            "augmentation": self.config["augmentation"],
            "feature_engineering": self.config["feature_engineering"]
        }
        
        self.preprocessing_pipeline = PreprocessingPipeline(
            preprocessing_config, self.logger
        )
        
        # Try to load existing processed data
        if not force_reprocess:
            processed_data = self.preprocessing_pipeline.load_processed_dataset()
            if processed_data is not None:
                self.logger.info("Using existing processed data")
                return processed_data
        
        # Process data from scratch
        self.logger.info("Processing data from scratch...")
        
        # Check if raw data exists
        raw_data_path = self.preprocessing_pipeline.raw_data_path
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data directory not found: {raw_data_path}")
        
        # Get list of words
        words = [d for d in os.listdir(raw_data_path) 
                if os.path.isdir(os.path.join(raw_data_path, d)) and not d.startswith('.')]
        
        if not words:
            raise ValueError(f"No word directories found in: {raw_data_path}")
        
        words.sort()  # Ensure consistent ordering
        self.logger.info(f"Found {len(words)} words: {words}")
        
        # Process full pipeline
        processed_data = self.preprocessing_pipeline.process_full_pipeline(
            words, save_processed=True
        )
        
        return processed_data
    
    def train_model(self, processed_data: Dict) -> Dict:
        """Train the classification model.
        
        Args:
            processed_data: Processed dataset from preprocessing pipeline
            
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting model training...")
        
        # Extract training data
        X_train = processed_data["X_train"]
        y_train = processed_data["y_train"]
        X_val = processed_data["X_val"]
        y_val = processed_data["y_val"]
        
        # Get number of classes
        num_classes = y_train.shape[1]
        
        # Initialize classifier
        model_config = {
            **self.config["model"],
            "num_classes": num_classes,
            "sequence_length": self.config["preprocessing"]["sequence_length"],
            "feature_dim": self.config["preprocessing"]["feature_dim"]
        }
        
        self.classifier = AttentionLSTMClassifier(model_config, self.logger)
        
        # Build model
        model = self.classifier.build_model()
        self.logger.info(f"Model architecture:\n{self.classifier.get_model_summary()}")
        
        # Create model save directory
        model_save_path = self.config["training"]["model_save_path"]
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Train model
        history = self.classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config["training"]["epochs"],
            batch_size=self.config["training"]["batch_size"],
            model_save_path=model_save_path
        )
        
        return history
    
    def evaluate_model(self, processed_data: Dict) -> Dict:
        """Evaluate trained model on test set.
        
        Args:
            processed_data: Processed dataset from preprocessing pipeline
            
        Returns:
            Evaluation results dictionary
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        self.logger.info("Evaluating model on test set...")
        
        X_test = processed_data["X_test"]
        y_test = processed_data["y_test"]
        
        evaluation_results = self.classifier.evaluate_model(X_test, y_test)
        
        # Print evaluation summary
        self.logger.info(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
        self.logger.info(f"Test Loss: {evaluation_results['loss']:.4f}")
        
        return evaluation_results
    
    def plot_training_history(self, history: Dict, save_path: str = "training_history.png") -> None:
        """Plot training history.
        
        Args:
            history: Training history from model training
            save_path: Path to save the plot
        """
        if not self.config["training"]["plot_history"]:
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        self.logger.info(f"Training history plot saved to: {save_path}")
    
    def save_training_results(self, processed_data: Dict, history: Dict, 
                            evaluation_results: Dict) -> None:
        """Save training results and metadata.
        
        Args:
            processed_data: Processed dataset
            history: Training history
            evaluation_results: Evaluation results
        """
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save training configuration
        with open(os.path.join(results_dir, "training_config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        
        # Save training history
        with open(os.path.join(results_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)
        
        # Save evaluation results
        with open(os.path.join(results_dir, "evaluation_results.json"), "w") as f:
            json.dump(evaluation_results, f, indent=4, default=str)
        
        # Save dataset statistics
        with open(os.path.join(results_dir, "dataset_statistics.json"), "w") as f:
            json.dump(processed_data["statistics"], f, indent=4)
        
        self.logger.info(f"Training results saved to: {results_dir}")
    
    def run_full_training(self, force_reprocess: bool = False) -> Dict:
        """Run complete training pipeline.
        
        Args:
            force_reprocess: Whether to force data reprocessing
            
        Returns:
            Complete training results dictionary
        """
        self.logger.info("Starting full training pipeline...")
        
        try:
            # 1. Prepare data
            processed_data = self.prepare_data(force_reprocess)
            
            # 2. Train model
            history = self.train_model(processed_data)
            
            # 3. Evaluate model
            evaluation_results = self.evaluate_model(processed_data)
            
            # 4. Plot training history
            self.plot_training_history(history)
            
            # 5. Save results
            self.save_training_results(processed_data, history, evaluation_results)
            
            # 6. Return complete results
            training_results = {
                "processed_data": processed_data,
                "training_history": history,
                "evaluation_results": evaluation_results,
                "config": self.config
            }
            
            self.logger.info("Full training pipeline completed successfully!")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main training function."""
    logger.info("Hand Sign Language Recognition - Training")
    logger.info("="*60)
    
    # Initialize training manager
    trainer = TrainingManager()
    
    # Run full training pipeline
    results = trainer.run_full_training(force_reprocess=False)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final test accuracy: {results['evaluation_results']['accuracy']:.4f}")

if __name__ == "__main__":
    main() 