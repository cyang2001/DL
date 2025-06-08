"""
Classification module for hand sign language recognition.

This module provides LSTM-based classification with attention mechanism.
"""

from .classifier import AttentionLSTMClassifier

__all__ = ['AttentionLSTMClassifier']
