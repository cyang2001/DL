"""
Author: @Chen YANG
Date: 2025-05-27
Description: This file contains the LSTM with attention mechanism for sign language classification.
"""

import logging
from typing import Dict
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda, LayerNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import TopKCategoricalAccuracy
import numpy as np
from utils import get_logger
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from typing import List, Tuple
class AttentionLSTMClassifier:
    """LSTM with attention mechanism for sign language classification."""
    
    def __init__(self, config_dict: Dict|None = None, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
        if config_dict is None:
            self.logger.info("No config dictionary provided, using default values.")
            config_dict = {
                "num_classes": 15,
                "sequence_length": 30,
                "feature_dim": 1662,
                "lstm_units_1": 64,
                "lstm_units_2": 48,
                "dense_units": 32,
                "dropout": 0.2,
                "l2_regularization": 0.01,
                "learning_rate": 0.0005,
                "confidence_threshold": 0.7,
                "smoothing_window": 5
            }
        else:
            self.logger.info(f"Using provided config dictionary: {config_dict}")
        self.num_classes = config_dict.get("num_classes")
        self.sequence_length = config_dict.get("sequence_length")
        self.feature_dim = config_dict.get("feature_dim")
        self.lstm_units_1 = config_dict.get("lstm_units_1")
        self.lstm_units_2 = config_dict.get("lstm_units_2")
        self.dense_units = config_dict.get("dense_units")
        self.dropout = config_dict.get("dropout")
        self.l2_regularization = config_dict.get("l2_regularization")
        self.learning_rate = config_dict.get("learning_rate")
        self.confidence_threshold = config_dict.get("confidence_threshold")
        self.smoothing_window = config_dict.get("smoothing_window")
        self.prediction_history = []

        self.logger.info(f"Initialized AttentionLSTMClassifier with config: {config_dict}")
    def attention_layer(self, lstm_output):
        """Custom attention layer."""
        # lstm_output shape: (batch_size, sequence_length, lstm_units)
        attention_weights = Dense(1, activation='tanh', name='attention_dense')(lstm_output)
        attention_weights = Flatten(name='attention_flatten')(attention_weights)
        attention_weights = Activation('softmax', name='attention_softmax')(attention_weights)
        attention_weights = RepeatVector(lstm_output.shape[-1], name='attention_repeat')(attention_weights)
        attention_weights = Permute([2, 1], name='attention_permute')(attention_weights)
        
        # Use attention weights to weight the LSTM output
        attended_output = Multiply(name='attention_multiply')([lstm_output, attention_weights])
        attended_output = Lambda(lambda x: tf.reduce_sum(x, axis=1), 
                               output_shape=(lstm_output.shape[-1],), 
                               name='attention_sum')(attended_output)
        
        return attended_output
    
    def build_model(self) -> Model:
        """Build LSTM model with attention mechanism."""
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        
        lstm1 = LSTM(
            self.lstm_units_1, 
            return_sequences=True, 
            dropout=self.dropout,
            kernel_regularizer=l2(self.l2_regularization),
            recurrent_dropout=0.1,
            name='lstm1')(inputs)
        lstm1 = LayerNormalization(name='layer_norm1')(lstm1)
        lstm2 = LSTM(
            self.lstm_units_2, 
            return_sequences=True, 
            dropout=self.dropout,
            kernel_regularizer=l2(self.l2_regularization),
            recurrent_dropout=0.1,
            name='lstm2')(lstm1)
        lstm2 = LayerNormalization(name='layer_norm2')(lstm2)
        
        attended_features = self.attention_layer(lstm2)
        
        dense1 = Dense(
            self.dense_units, 
            activation='relu', 
            kernel_regularizer=l2(self.l2_regularization), 
            name='dense1')(attended_features)
        dropout1 = Dropout(self.dropout, name='dropout1')(dense1)
        
        outputs = Dense(
            self.num_classes, 
            activation='softmax', 
            name='output_layer')(dropout1)
        
        model = Model(inputs=inputs, outputs=outputs, name='AttentionLSTM')
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        self.model = model
        self.logger.info(f"Model built successfully with {model.count_params()} parameters")
        return model
    
    def get_callbacks(self, model_save_path: str = "best_model.h5"):
        """Get training callbacks for optimal training.
        
        Args:
            model_save_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=30,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 200, batch_size: int = 16,
              model_save_path: str = "best_model.h5") -> Dict:
        """Train the attention LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels (one-hot encoded)
            X_val: Validation sequences  
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Training batch size
            model_save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
            
        self.logger.info(f"Starting training with {len(X_train)} training samples")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_save_path),
            verbose=1
        )
        
        self.is_trained = True
        self.logger.info("Training completed successfully")
        return history.history
    
    def predict_single(self, sequence: np.ndarray, 
                      class_names: List[str] = None) -> Tuple[str, float, np.ndarray]:
        """Predict a single sequence with confidence scoring.
        
        Args:
            sequence: Input sequence with shape (30, 1662)
            class_names: List of class names for interpretation
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        if not self.is_trained:
            self.logger.warning("Model may not be trained yet.")
            
        if sequence.shape != (self.sequence_length, self.feature_dim):
            raise ValueError(f"Expected shape ({self.sequence_length}, {self.feature_dim}), got {sequence.shape}")
        
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        probabilities = self.model.predict(sequence_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        if class_names is not None:
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"class_{predicted_class_idx}"
            
        return predicted_class, confidence, probabilities
    
    def predict_realtime(self, sequence: np.ndarray, 
                        class_names: List[str] = None) -> Tuple[str, float, bool]:
        """Real-time prediction with smoothing and confidence filtering.
        
        Args:
            sequence: Input sequence with shape (30, 1662)
            class_names: List of class names for interpretation
            
        Returns:
            Tuple of (predicted_class, confidence, is_confident)
        """
        predicted_class, confidence, probabilities = self.predict_single(sequence, class_names)
        
        self.prediction_history.append({
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        })
        
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)
        
        if len(self.prediction_history) >= 3:
            smoothed_class, smoothed_confidence = self._smooth_predictions()
            is_confident = smoothed_confidence >= self.confidence_threshold
            return smoothed_class, smoothed_confidence, is_confident
        else:
            is_confident = confidence >= self.confidence_threshold
            return predicted_class, confidence, is_confident
    
    def _smooth_predictions(self) -> Tuple[str, float]:
        """Smooth predictions using recent history.
        
        Returns:
            Tuple of (smoothed_class, smoothed_confidence)
        """
        if not self.prediction_history:
            return "unknown", 0.0
        
        # Calculate weighted average probability for each class
        recent_predictions = self.prediction_history[-3:]  # Recent 3 predictions
        weights = np.array([0.2, 0.3, 0.5])  # Higher weight for recent predictions
        
        # Weighted average probability
        weighted_probs = np.zeros(self.num_classes)
        for i, pred in enumerate(recent_predictions):
            weighted_probs += weights[i] * pred['probabilities']
        
        # Get final prediction
        predicted_idx = np.argmax(weighted_probs)
        confidence = weighted_probs[predicted_idx]
        
        # Get class name
        if self.prediction_history[-1].get('class'):
            # Get class name from history
            recent_class = self.prediction_history[-1]['class']
            if recent_class.startswith('class_'):
                predicted_class = f"class_{predicted_idx}"
            else:
                #TODO
                # Assume there is a list of class names
                predicted_class = f"class_{predicted_idx}"  # Adjust this based on actual class names
        else:
            predicted_class = f"class_{predicted_idx}"
        
        return predicted_class, confidence
    
    def reset_prediction_history(self):
        """Reset prediction history for new session."""
        self.prediction_history = []
        self.logger.info("Prediction history reset")
    
    def save_model(self, filepath: str):
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Build model first.")
        
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = load_model(filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            return "Model not built yet."
        
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance.
        
        Args:
            X_test: Test sequences
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        evaluation_results = {
            'loss': results[0],
            'accuracy': results[1],
            'top_3_accuracy': results[2] if len(results) > 2 else None,
            'classification_report': classification_report(y_true_classes, y_pred_classes),
            'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes).tolist()
        }
        
        self.logger.info(f"Model evaluation completed. Accuracy: {results[1]:.4f}")
        return evaluation_results