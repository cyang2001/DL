"""
Author: @Chen YANG
Date: 2025-06-08
Offline demo script to verify model performance on saved sequences.

The script loads:
1. Processed sequences & labels from `processed_data/`
2. Trained model from `models/best_hand_sign_model.h5`
Then performs a quick prediction on a random subset and reports accuracy
and a confusion matrix.  Useful to check whether the lack of response in
`demo.py` comes from model or from real-time data capture.
"""

import os
import json
import random
import numpy as np
from typing import List

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils import get_logger
from src.preprocessing import PreprocessingPipeline
from src.classification import AttentionLSTMClassifier

logger = get_logger(__name__)

DATA_DIR = "processed_data"
MODEL_PATH = "models/best_hand_sign_model.h5"
SAMPLE_SIZE = None  

SEQUENCE_LENGTH = 30
FEATURE_DIM = 1662

def load_dataset():
    """Load processed dataset saved by preprocessing pipeline."""
    seq_path = os.path.join(DATA_DIR, "sequences.npy")
    lbl_path = os.path.join(DATA_DIR, "labels.npy")
    meta_path = os.path.join(DATA_DIR, "metadata.json")

    if not (os.path.exists(seq_path) and os.path.exists(lbl_path)):
        logger.error("Processed sequences/labels not found. Run preprocessing first.")
        exit(1)

    sequences = np.load(seq_path)  # shape (N,30,1662)
    labels_int = np.load(lbl_path)  # integer labels (N,)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    word_to_idx = meta["word_to_idx"]
    idx_to_word = {idx: w for w, idx in word_to_idx.items()}

    return sequences, labels_int, idx_to_word


def init_preprocessor():
    cfg = {
        "raw_data_path": "",  # not used
        "processed_data_path": DATA_DIR,
        "preprocessing": {
            "sequence_length": SEQUENCE_LENGTH,
            "feature_dim": FEATURE_DIM,
            "enable_normalization": True,
            "normalization_method": "standard",
            "enable_smoothing": True,
            "smoothing_window": 3,
            "enable_interpolation": True,
            "enable_quality_check": False,
        },
        "augmentation": {"enable_augmentation": False},
        "feature_engineering": {
            "enable_feature_engineering": True,
            "extract_velocity": True,
            "extract_acceleration": True,
            "extract_angles": True,
            "extract_distances": True,
        },
    }
    return PreprocessingPipeline(cfg, logger)


def init_classifier(num_classes: int, feature_dim: int):
    cfg = {
        "num_classes": num_classes,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": feature_dim,
        "lstm_units_1": 64,
        "lstm_units_2": 48,
        "dense_units": 32,
        "dropout": 0.2,
        "l2_regularization": 0.01,
        "learning_rate": 0.0005,
        "confidence_threshold": 0.7,
        "smoothing_window": 5,
    }
    clf = AttentionLSTMClassifier(cfg, logger)
    clf.load_model(MODEL_PATH)
    return clf


def main():
    logger.info("Offline test demo – start")

    sequences, labels_int, idx_to_word = load_dataset()

    # Build class_names list ordered by label index (0…N-1)
    global CLASS_NAMES  # update module-level constant for downstream functions
    CLASS_NAMES = [idx_to_word[idx] for idx in range(len(idx_to_word))]

    num_classes = len(CLASS_NAMES)

    logger.info("Label mapping (index → word): " + str({i: w for i, w in enumerate(CLASS_NAMES)}))

    # Select subset for quick test
    indices = list(range(len(sequences)))
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(indices):
        indices = random.sample(indices, SAMPLE_SIZE)

    sequences_sel = sequences[indices]
    labels_sel = labels_int[indices]

    preprocessor = init_preprocessor()
    # Load existing scaler
    preprocessor.load_processed_dataset()

    # If scaler still not fitted (e.g., scaler.pkl missing), fit on engineered full dataset
    if not hasattr(preprocessor.data_preprocessor.scaler, "mean_"):
        logger.warning("Scaler missing, fitting on entire dataset for this test run")

        # Apply same feature engineering as used during training
        if preprocessor.feature_engineer.enable_feature_engineering:
            sequences_fe = np.array([preprocessor.feature_engineer.extract_features(s) for s in sequences])
        else:
            sequences_fe = sequences

        preprocessor.data_preprocessor.normalize_sequences(sequences_fe, fit_scaler=True)

    # Infer engineered feature dimension using first sequence
    sample_engineered = preprocessor.feature_engineer.extract_features(sequences_sel[0])
    feature_dim = sample_engineered.shape[1]

    classifier = init_classifier(num_classes, feature_dim)

    y_true = []
    y_pred = []

    for seq, label_int in zip(sequences_sel, labels_sel):
        seq_proc = preprocessor.preprocess_single_sequence(seq)
        word_pred, conf, _ = classifier.predict_single(seq_proc, class_names=CLASS_NAMES)
        y_pred.append(word_pred)
        y_true.append(idx_to_word[label_int])
        logger.info(f"GT: {idx_to_word[label_int]:<10} | Pred: {word_pred:<10} | conf={conf:.2f}")

    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Overall accuracy on {len(y_true)} samples: {acc*100:.2f}%")
    logger.info("Classification report:\n" + classification_report(y_true, y_pred))
    logger.info("Confusion matrix:\n" + np.array2string(confusion_matrix(y_true, y_pred)))


if __name__ == "__main__":
    main()
