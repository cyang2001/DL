
"""
Author: @De oliveira Léna & @Sron Sarah
Date: 04/06/2025
Feature engineering module for hand sign language recognition.

This module provides advanced feature extraction capabilities including
velocity, acceleration, angles, and other derived features.

"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from utils import get_logger


class FeatureEngineer:
    """Advanced feature engineering for hand sign language sequences."""
    
    def __init__(self, config_dict: Union[Dict, None] = None, logger: logging.Logger = None):
        """Initialize the feature engineer.
        
        Args:
            config_dict: Configuration dictionary containing feature engineering parameters
            logger: Logger instance for recording operations
        """
        self.logger = logger or get_logger(__name__)
        
        if config_dict is None:
            self.logger.info("No config dictionary provided, using default values.")
            config_dict = {
                "enable_feature_engineering": False,
                "extract_velocity": True,
                "extract_acceleration": True,
                "extract_angles": True,
                "extract_distances": True,
                "extract_relative_positions": True,
                "extract_statistical_features": False,
                "statistical_window_size": 5,
                "extract_hand_shape_features": True,
                "extract_hand_orientation": True,
                "extract_pose_angles": True
            }
        else:
            self.logger.info(f"Using provided config dictionary: {config_dict}")
            
        # Initialize configuration parameters
        self.enable_feature_engineering = config_dict.get("enable_feature_engineering", False)
        self.extract_velocity = config_dict.get("extract_velocity", True)
        self.extract_acceleration = config_dict.get("extract_acceleration", True)
        self.extract_angles = config_dict.get("extract_angles", True)
        self.extract_distances = config_dict.get("extract_distances", True)
        self.extract_relative_positions = config_dict.get("extract_relative_positions", True)
        self.extract_statistical_features = config_dict.get("extract_statistical_features", False)
        self.statistical_window_size = config_dict.get("statistical_window_size", 5)
        self.extract_hand_shape_features = config_dict.get("extract_hand_shape_features", True)
        self.extract_hand_orientation = config_dict.get("extract_hand_orientation", True)
        self.extract_pose_angles = config_dict.get("extract_pose_angles", True)
        
        # Feature dimensions (MediaPipe structure)
        self.pose_dim = 132  # 33 points × 4 (x,y,z,visibility)
        self.face_dim = 1404  # 468 points × 3 (x,y,z)
        self.hand_dim = 63   # 21 points × 3 (x,y,z)
        
        # Key point indices for hands
        self.hand_landmarks = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20,
            'thumb_mcp': 2,
            'index_mcp': 5,
            'middle_mcp': 9,
            'ring_mcp': 13,
            'pinky_mcp': 17
        }
        
        self.logger.info(f"FeatureEngineer initialized with config: {config_dict}")
        
    def extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract advanced features from a sequence.
        
        Args:
            sequence: Input sequence with shape (sequence_length, feature_dim)
            
        Returns:
            Enhanced sequence with additional features
        """
        # 1. Check if feature engineering is enabled
        if not self.enable_feature_engineering:
            return sequence

        # 2. Start with original features
        features = [sequence]

        # 3. Extract features based on config
        if self.extract_velocity:
            features.append(self._extract_velocity_features(sequence))
        if self.extract_acceleration:
            features.append(self._extract_acceleration_features(sequence))
        if self.extract_angles:
            features.append(self._extract_angle_features(sequence))
        if self.extract_distances:
            features.append(self._extract_distance_features(sequence))
        if self.extract_relative_positions:
            features.append(self._extract_relative_position_features(sequence))
        if self.extract_statistical_features:
            features.append(self._extract_statistical_features(sequence))
        if self.extract_hand_shape_features:
            features.append(self._extract_hand_shape_features(sequence))
        if self.extract_hand_orientation:
            features.append(self._extract_hand_orientation_features(sequence))
        if self.extract_pose_angles:
            features.append(self._extract_pose_angle_features(sequence))

        # 4. Concatenate all features along the last axis
        enhanced_sequence = np.concatenate(features, axis=-1)

        # 5. Return enhanced sequence
        return enhanced_sequence

        
    def _extract_velocity_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract velocity features from the sequence.

        Args:
            sequence: Input sequence

        Returns:
            Velocity features
        """
        # 1. Calculate velocity as difference between consecutive frames
        velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])

        # 2. Only calculate for non-zero coordinates
        # Créer un masque indiquant où la différence est valide (aucun des deux n’est zéro)
        valid_mask = (sequence != 0) & (np.roll(sequence, shift=1, axis=0) != 0)
        velocity = velocity * valid_mask

        # 3. Return velocity features
        return velocity


        
    def _extract_acceleration_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract acceleration features from the sequence.

        Args:
            sequence: Input sequence

        Returns:
            Acceleration features
        """
        # 1. First extract velocity
        velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
        valid_velocity_mask = (sequence != 0) & (np.roll(sequence, shift=1, axis=0) != 0)
        velocity = velocity * valid_velocity_mask

        # 2. Then extract acceleration as velocity changes
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        valid_accel_mask = (velocity != 0) & (np.roll(velocity, shift=1, axis=0) != 0)
        acceleration = acceleration * valid_accel_mask

        # 3. Return acceleration features
        return acceleration

        
    def _extract_angle_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract angle features between key points.

        Args:
            sequence: Input sequence

        Returns:
            Angle features
        """
        # 1. Extract hand coordinates for both hands
        left_hand = self._extract_hand_coordinates(sequence, 'left')
        right_hand = self._extract_hand_coordinates(sequence, 'right')
        sequence_length = sequence.shape[0]
        angles = []

        for i in range(sequence_length):
            frame_angles = []

            # 2. Calculate angles between key points for each hand
            for hand_data in [left_hand, right_hand]:
                if hand_data is not None:
                    try:
                        # Angle between thumb_mcp - wrist - index_mcp
                        p1 = hand_data[i, self.hand_landmarks['thumb_mcp']]
                        p2 = hand_data[i, self.hand_landmarks['wrist']]
                        p3 = hand_data[i, self.hand_landmarks['index_mcp']]
                        angle1 = self._calculate_angle(p1, p2, p3)
                        frame_angles.append(angle1)

                        # Angle between index_mcp - wrist - pinky_mcp
                        p1 = hand_data[i, self.hand_landmarks['index_mcp']]
                        p3 = hand_data[i, self.hand_landmarks['pinky_mcp']]
                        angle2 = self._calculate_angle(p1, p2, p3)
                        frame_angles.append(angle2)

                        # Autres angles entre doigts
                        for mcp1, mcp2 in [('index_mcp', 'middle_mcp'), ('middle_mcp', 'ring_mcp'), ('ring_mcp', 'pinky_mcp')]:
                            p1 = hand_data[i, self.hand_landmarks[mcp1]]
                            p3 = hand_data[i, self.hand_landmarks[mcp2]]
                            angle = self._calculate_angle(p1, p2, p3)
                            frame_angles.append(angle)

                        # Total = 5 angles par main
                    except:
                        frame_angles.extend([0.0] * 5)
                else:
                    frame_angles.extend([0.0] * 5)

            # 3. Return angle features
            angles.append(frame_angles)

        return np.array(angles)

        
    def _extract_distance_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract distance features between key points.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Distance features
        """
        # TODO: Implement distance extraction
        # 1. Calculate hand-to-hand distances
        # 2. Calculate intra-hand distances
        # 3. Return distance features

        sequence_length = sequence.shape[0]
        distance_features = []

        for frame in sequence:
            # 1. Reshape frame to (num_keypoints, coords), assuming 3D keypoints
            num_keypoints = frame.shape[0] // 3
            keypoints = frame.reshape(num_keypoints, 3)

            # Split hands
            left_hand = keypoints[468:489]    # 21 keypoints
            right_hand = keypoints[522:543]   # 21 keypoints

            # Hand-to-hand distances (e.g., palm centers and fingertips)
            hand_distances = []
            if left_hand.shape[0] >= 1 and right_hand.shape[0] >= 1:
                hand_distances.append(np.linalg.norm(left_hand[0] - right_hand[0]))  # center to center
                for i in [4, 8, 12, 16, 20]:  # fingertips
                    hand_distances.append(np.linalg.norm(left_hand[i] - right_hand[i]))

            # Intra-hand distances (left hand)
            left_distances = []
            for i in range(1, 21):
                left_distances.append(np.linalg.norm(left_hand[0] - left_hand[i % 21]))

            # Intra-hand distances (right hand)
            right_distances = []
            for i in range(1, 21):
                right_distances.append(np.linalg.norm(right_hand[0] - right_hand[i % 21]))

            # Concat all
            distances = hand_distances[:5] + left_distances[:10] + right_distances[:10]
            distance_features.append(distances)

        return np.array(distance_features)


        
    def _extract_relative_position_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract relative position features.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Relative position features
        """
        # 1. Extract pose key points (shoulders, nose)
        left_shoulder = sequence[:, 502:505]
        right_shoulder = sequence[:, 523:526]
        nose = sequence[:, 478:481]
        
        # 2. Calculate relative positions of hands to body parts
        left_hand = sequence[:, 0:3]
        right_hand = sequence[:, 21*3:21*3+3]
        
        rel_left_to_nose = left_hand - nose
        rel_right_to_nose = right_hand - nose
        rel_left_to_shoulder = left_hand - left_shoulder
        rel_right_to_shoulder = right_hand - right_shoulder

        # 3. Return relative position features
        relative_features = np.concatenate([
            rel_left_to_nose, rel_right_to_nose,
            rel_left_to_shoulder, rel_right_to_shoulder
        ], axis=1)  # shape (T, 12)
        
        return relative_features

        
    def _extract_statistical_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract statistical features over sliding windows.

        Args:
            sequence: Input sequence

        Returns:
            Statistical features
        """
        # 1. Use sliding window approach
        window_size = self.statistical_window_size if hasattr(self, 'statistical_window_size') else 5
        pad = window_size // 2
        padded_seq = np.pad(sequence, ((pad, pad), (0, 0)), mode='edge')

        features = []

        # 2. Calculate statistics (mean, std, range) for each window
        for i in range(sequence.shape[0]):
            window = padded_seq[i:i + window_size]
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            rng = np.ptp(window, axis=0)  # ptp = max - min
            features.append(np.concatenate([mean, std, rng]))

        # 3. Return statistical features
        return np.stack(features)

        
    def _extract_hand_shape_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract hand shape features.

        Args:
            sequence: Input sequence

        Returns:
            Hand shape features
        """
        # 1. Extract hand coordinates
        left_hand_indices = list(range(468, 489))  # 21 points
        right_hand_indices = list(range(522, 543))  # 21 points

        left_coords = sequence[:, np.array([[3*i, 3*i+1, 3*i+2] for i in left_hand_indices]).flatten()]
        right_coords = sequence[:, np.array([[3*i, 3*i+1, 3*i+2] for i in right_hand_indices]).flatten()]

        left_coords = left_coords.reshape(sequence.shape[0], 21, 3)
        right_coords = right_coords.reshape(sequence.shape[0], 21, 3)

        # 2. Calculate shape descriptors (span, distances, etc.)
        def shape_descriptors(hand_coords):
            span = np.linalg.norm(hand_coords[:, 0] - hand_coords[:, -1], axis=1)  # palm base to tip
            max_dist = np.max(np.linalg.norm(
                hand_coords[:, :, np.newaxis, :] - hand_coords[:, np.newaxis, :, :], axis=-1
            ), axis=(1, 2))
            mean_dist = np.mean(np.linalg.norm(
                hand_coords[:, :, np.newaxis, :] - hand_coords[:, np.newaxis, :, :], axis=-1
            ), axis=(1, 2))
            std_dist = np.std(np.linalg.norm(
                hand_coords[:, :, np.newaxis, :] - hand_coords[:, np.newaxis, :, :], axis=-1
            ), axis=(1, 2))
            return np.stack([span, max_dist, mean_dist, std_dist], axis=1)

        left_features = shape_descriptors(left_coords)
        right_features = shape_descriptors(right_coords)

        # Add more if needed to reach 15 features (ex: bounding box size, number of fingers extended…)
        # For now we'll pad with zeros to get 15 per hand
        left_padded = np.pad(left_features, ((0, 0), (0, 11)), mode='constant')
        right_padded = np.pad(right_features, ((0, 0), (0, 11)), mode='constant')

        # 3. Return shape features
        return np.concatenate([left_padded, right_padded], axis=1)

        
    def _extract_hand_orientation_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract hand orientation features.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Hand orientation features
        """
        # 1. Calculate hand direction vectors
        left_hand = sequence[:, 444:465].reshape(-1, 7, 3)
        right_hand = sequence[:, 681:702].reshape(-1, 7, 3)

        # Direction vector: wrist to middle finger
        left_vectors = left_hand[:, 3] - left_hand[:, 0]
        right_vectors = right_hand[:, 3] - right_hand[:, 0]

        # 2. Compute orientation angles (with respect to x, y, z axes)
        def compute_angles(v):
            norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            unit = v / norm
            angles = np.arccos(np.clip(unit, -1.0, 1.0))  # angles to x, y, z
            return angles

        left_angles = compute_angles(left_vectors)
        right_angles = compute_angles(right_vectors)

        # 3. Return orientation features
        return np.hstack([left_angles, right_angles])

        
    def _extract_pose_angle_features(self, sequence: np.ndarray) -> np.ndarray: 
        """Extract pose angle features.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Pose angle features
        """
        # TODO: Implement pose angle extraction
        # 1. Define key pose connections
        # 2. Calculate angles for each connection
        # 3. Return pose angle features

        sequence_length = sequence.shape[0]

        # Extrait les coordonnées des points clés (ex : épaule, coude, poignet)
        def get_point_coords(seq, point_index):
            start = point_index * 3
            return seq[:, start:start+3]

        # Exemple : connections épaule-coude-poignet
        left_shoulder = get_point_coords(sequence, 11)
        left_elbow = get_point_coords(sequence, 13)
        left_wrist = get_point_coords(sequence, 15)

        right_shoulder = get_point_coords(sequence, 12)
        right_elbow = get_point_coords(sequence, 14)
        right_wrist = get_point_coords(sequence, 16)

        def angle_between(a, b, c):
            ba = a - b
            bc = c - b
            dot_product = np.sum(ba * bc, axis=1)
            norm_ba = np.linalg.norm(ba, axis=1)
            norm_bc = np.linalg.norm(bc, axis=1)
            cos_angle = np.clip(dot_product / (norm_ba * norm_bc + 1e-6), -1.0, 1.0)
            return np.arccos(cos_angle)

        left_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = angle_between(right_shoulder, right_elbow, right_wrist)

        # Exemple : angle entre épaules et hanches (axe du torse)
        left_hip = get_point_coords(sequence, 23)
        right_hip = get_point_coords(sequence, 24)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        torso_angle = angle_between(left_shoulder, shoulder_center, hip_center)

        # Placeholder : on ajoute deux angles fixes à zéro pour compléter les 5
        zero_angle = np.zeros(sequence_length)

        return np.stack([
            left_elbow_angle,
            right_elbow_angle,
            torso_angle,
            zero_angle,
            zero_angle
        ], axis=1)

        
    def _extract_hand_coordinates(self, sequence: np.ndarray, hand: str) -> Optional[np.ndarray]:
        """Extract hand coordinates from sequence.

        Args:
            sequence: Input sequence
            hand: 'left' or 'right'

        Returns:
            Hand coordinates or None if not available
        """
        # 1. Calculate start and end indices based on hand
        if hand == 'left':
            start_idx = 0
        elif hand == 'right':
            start_idx = 63
        else:
            return None  # Hand not recognized

        end_idx = start_idx + 63  # 21 keypoints × 3D (x,y,z)

        # 2. Extract hand data from sequence
        hand_data = sequence[:, start_idx:end_idx]

        # 3. Check if hand data is available
        if np.all(hand_data == 0):
            return None

        # 4. Reshape to (sequence_length, 21, 3)
        hand_coordinates = hand_data.reshape(sequence.shape[0], 21, 3)

        # 5. Return hand coordinates or None
        return hand_coordinates

        
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle formed by three points.

        Args:
            p1, p2, p3: Three points in 3D space

        Returns:
            Angle in radians
        """
        # 1. Create vectors from p2 to p1 and p2 to p3
        v1 = p1 - p2
        v2 = p3 - p2

        # 2. Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

        # 3. Calculate angle using dot product
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # avoid numerical errors
        angle = np.arccos(dot_product)

        # 4. Return angle in radians
        return angle
