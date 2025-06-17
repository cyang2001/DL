"""
Author: @Chen YANG
Date: 2025-06-08
Real-time hand sign language recognition demo.

This script loads the trained Attention LSTM model and performs real-time
inference from the webcam. Detected words are rendered on the video feed and
concatenated into a sentence.
"""

import os
import sys
import logging
from typing import List

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model

from utils import get_logger, extract_keypoints
from src.preprocessing import PreprocessingPipeline
from src.classification import AttentionLSTMClassifier


SEQUENCE_LENGTH = 30
FEATURE_DIM = 1662


CLASS_NAMES: List[str] = [
    "class",       
    "deep",         
    "for",          
    "learning",     
    "thank you",    
    "this"         
]

MODEL_PATH = "models/best_hand_sign_model.h5"


PREPROCESSING_CONFIG = {
    "sequence_length": SEQUENCE_LENGTH,
    "feature_dim": FEATURE_DIM,
    "enable_normalization": True,
    "normalization_method": "standard",
    "enable_smoothing": True,
    "smoothing_window": 3,
    "enable_interpolation": True,
    "enable_quality_check": False  
}

logger = get_logger(__name__)


def _initialize_classifier() -> AttentionLSTMClassifier:
    """Load the trained model and return a classifier instance.

    Returns
    -------
    AttentionLSTMClassifier
        Loaded classifier ready for inference.
    """
    cfg = {
        "num_classes": len(CLASS_NAMES),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,

        "lstm_units_1": 64,
        "lstm_units_2": 48,
        "dense_units": 32,
        "dropout": 0.2,
        "l2_regularization": 0.01,
        "learning_rate": 0.0005,
        "confidence_threshold": 0.7,
        "smoothing_window": 5,
    }

    classifier = AttentionLSTMClassifier(cfg, logger)
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)

    classifier.load_model(MODEL_PATH)
    logger.info("Classifier loaded successfully")
    return classifier


def _initialize_preprocessor() -> PreprocessingPipeline:
    """Create a preprocessing pipeline for single-sequence inference."""
    pipeline_cfg = {
        "raw_data_path": "",  # not used
        "processed_data_path": "",  # not used
        "preprocessing": PREPROCESSING_CONFIG,
        "augmentation": {"enable_augmentation": False},
        "feature_engineering": {"enable_feature_engineering": False},
    }
    return PreprocessingPipeline(pipeline_cfg, logger)




def run_realtime_demo() -> None:
    """Start webcam and perform real-time hand sign language recognition."""
    logger.info("Initializing real-time demo …")


    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False,
                                    model_complexity=1,
                                    smooth_landmarks=True,
                                    enable_segmentation=False,
                                    refine_face_landmarks=False)

    preprocessor = _initialize_preprocessor()
    classifier = _initialize_classifier()

    sequence_frames: List[np.ndarray] = []
    recognized_sentence: List[str] = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        sys.exit(1)

    logger.info("Webcam opened. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                break

            frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            keypoints = extract_keypoints(results)  # shape (1662,)

            sequence_frames.append(keypoints)
            if len(sequence_frames) > SEQUENCE_LENGTH:
                sequence_frames.pop(0)

            if len(sequence_frames) == SEQUENCE_LENGTH:
                sequence_np = np.array(sequence_frames)  # (30,1662)
                sequence_processed = preprocessor.preprocess_single_sequence(sequence_np)

                pred_word, conf, is_conf = classifier.predict_realtime(
                    sequence_processed, class_names=CLASS_NAMES)

                if is_conf:
                    if len(recognized_sentence) == 0 or recognized_sentence[-1] != pred_word:
                        recognized_sentence.append(pred_word)
                        logger.info(f"Detected word: {pred_word} (conf={conf:.2f})")

            curr_word_text = recognized_sentence[-1] if recognized_sentence else "…"
            sentence_text = " ".join(recognized_sentence[-10:])  # show last 10 words

            cv2.putText(frame, f"Current: {curr_word_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Sentence: {sentence_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Hand Sign Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        logger.info("Demo terminated.")



if __name__ == "__main__":
    run_realtime_demo()
