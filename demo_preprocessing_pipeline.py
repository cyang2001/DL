"""
Author: @Chen YANG
Date: 2025-06-02
Demo script showing how to use the preprocessing pipeline.

This script demonstrates the usage of the preprocessing modules
for hand sign language recognition data.

"""

import os
from utils import get_logger
from src.preprocessing import PreprocessingPipeline, DataPreprocessor, DataAugmentor, FeatureEngineer

logger = get_logger(__name__)

def demo_preprocessing_pipeline():
    """Demonstrate the complete preprocessing pipeline."""
    
    # Configuration for the preprocessing pipeline
    # Note: All paths are relative to project root directory
    config = {
        "raw_data_path": "MP_data",
        "processed_data_path": "processed_data",
        "validation_split": 0.2,
        "test_split": 0.1,
        "random_seed": 42,
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
            "enable_feature_engineering": False,  # Start with basic features
            "extract_velocity": True,
            "extract_acceleration": True,
            "extract_angles": True,
            "extract_distances": True
        }
    }
    
    # Initialize the preprocessing pipeline
    logger.info("Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline(config)
    
    # The pipeline will automatically resolve paths relative to project root
    # Check if raw data directory exists
    if not os.path.exists(pipeline.raw_data_path):
        logger.error(f"Raw data directory not found: {pipeline.raw_data_path}")
        logger.error("Please make sure you have collected data using collection.py first")
        logger.error("Run 'python collection.py' from the project root directory")
        return
    
    words = [d for d in os.listdir(pipeline.raw_data_path) 
             if os.path.isdir(os.path.join(pipeline.raw_data_path, d)) and not d.startswith('.')]
    
    if not words:
        logger.error(f"No word directories found in: {pipeline.raw_data_path}")
        logger.error("Please collect some data first using collection.py")
        return
    
    logger.info(f"Found {len(words)} words: {words}")
    
    logger.info("Checking for existing processed data...")
    existing_data = pipeline.load_processed_dataset()
    
    if existing_data is not None:
        logger.info("Found existing processed data. Using cached version.")
        logger.info(f"Dataset shape: {existing_data['X_train'].shape}")
    else:
        logger.info("No existing processed data found. Starting preprocessing...")
        
        try:
            processed_data = pipeline.process_full_pipeline(words, save_processed=True)
            
            logger.info("Preprocessing completed successfully!")
            logger.info(f"Training data shape: {processed_data['X_train'].shape}")
            logger.info(f"Validation data shape: {processed_data['X_val'].shape}")
            logger.info(f"Test data shape: {processed_data['X_test'].shape}")
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            logger.error("This is expected as the core implementation needs to be completed")

def demo_individual_components():
    """Demonstrate individual preprocessing components."""
    
    logger.info("\n" + "="*50)
    logger.info("Demonstrating individual preprocessing components")
    logger.info("="*50)
    
    logger.info("\n1. Data Preprocessor Demo:")
    preprocessor_config = {
        "sequence_length": 30,
        "feature_dim": 1662,
        "enable_normalization": True,
        "normalization_method": "standard"
    }
    
    preprocessor = DataPreprocessor(preprocessor_config)
    logger.info(f"Preprocessor initialized with sequence length: {preprocessor.sequence_length}")
    
    logger.info("\n2. Data Augmentor Demo:")
    augmentor_config = {
        "enable_augmentation": True,
        "augmentation_probability": 0.5,
        "enable_noise": True,
        "enable_time_warping": True
    }
    
    augmentor = DataAugmentor(augmentor_config, logger)
    logger.info(f"Augmentor initialized with probability: {augmentor.augmentation_probability}")
    
    logger.info("\n3. Feature Engineer Demo:")
    feature_config = {
        "enable_feature_engineering": False,
        "extract_velocity": True,
        "extract_acceleration": True
    }
    
    feature_engineer = FeatureEngineer(feature_config, logger)
    logger.info(f"Feature Engineer initialized. Engineering enabled: {feature_engineer.enable_feature_engineering}")

def main():
    """Main function to run all demos."""
    logger.info("Hand Sign Language Recognition - Preprocessing Demo")
    logger.info("="*60)
    logger.info("IMPORTANT: Run this script from the project root directory!")
    logger.info("="*60)
    
    demo_preprocessing_pipeline()
    
    demo_individual_components()
    
    logger.info("\n" + "="*60)
    logger.info("Demo completed!")

if __name__ == "__main__":
    main() 