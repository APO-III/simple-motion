"""
Real-time Motion Activity Classifier using XGBoost.

This application:
1. Captures frames from webcam in real-time
2. Extracts pose landmarks using MediaPipe
3. Calculates motion features
4. Classifies activity using trained XGBoost model
5. Displays statistics and predictions on screen

Usage:
    python realtime_motion_classifier.py
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque, Counter
from typing import Optional, Dict
import pickle
from sklearn.preprocessing import StandardScaler

from pose_processing.use_cases.pose_extraction_service import PoseExtractionService
from data_features.use_cases.features_service import FeaturesService
from ml_training.models.xgboost_model import XGBoostMotionClassifier


class RealTimeMotionClassifier:
    """
    Real-time motion activity classifier using webcam.
    """

    # MediaPipe Pose landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(
        self,
        model_path: str = "output/xgboost/xgboost_classifier.pkl",
        scaler_path: str = "output/xgboost/scaler.pkl",
        label_mappings_path: str = "output/xgboost/label_mappings.json",
        camera_index: int = 0,
        history_size: int = 30
    ):
        """
        Initialize real-time classifier.

        Args:
            model_path: Path to trained XGBoost model
            scaler_path: Path to feature scaler
            label_mappings_path: Path to label mappings JSON
            camera_index: Webcam index (default: 0)
            history_size: Number of recent predictions to track
        """
        self.camera_index = camera_index
        self.history_size = history_size
        self.prediction_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize services
        self.pose_extraction_service = PoseExtractionService(mediapipe_style=1)
        self.features_service = FeaturesService()
        
        # Load model and scaler
        print("Loading XGBoost model...")
        self.model = XGBoostMotionClassifier.load(model_path)
        print(f"Model loaded. Classes: {len(self.model.index_to_label)}")
        
        print("Loading feature scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded.")
        
        # Load label mappings
        import json
        with open(label_mappings_path, 'r') as f:
            label_mappings = json.load(f)
        self.index_to_label = {int(k): v for k, v in label_mappings['index_to_label'].items()}
        self.label_to_index = label_mappings['label_to_index']
        
        # Feature columns (must match training)
        self.feature_columns = [
            'normalized_leg_length',
            'shoulder_vector_x',
            'shoulder_vector_z',
            'ankle_vector_x',
            'ankle_vector_z',
            'average_hip_angle',
            'average_knee_angle'
        ]
        
        print("\n" + "=" * 70)
        print("Real-time Motion Classifier Ready!")
        print("=" * 70)
        print("Press 'q' to quit")
        print("=" * 70 + "\n")

    def _extract_landmark_coordinates(
        self,
        landmarks
    ) -> Dict[str, np.ndarray]:
        """Extract specific landmark coordinates as numpy arrays."""
        return {
            'left_shoulder': np.array([
                landmarks[self.LEFT_SHOULDER].x,
                landmarks[self.LEFT_SHOULDER].y,
                landmarks[self.LEFT_SHOULDER].z
            ]),
            'right_shoulder': np.array([
                landmarks[self.RIGHT_SHOULDER].x,
                landmarks[self.RIGHT_SHOULDER].y,
                landmarks[self.RIGHT_SHOULDER].z
            ]),
            'left_hip': np.array([
                landmarks[self.LEFT_HIP].x,
                landmarks[self.LEFT_HIP].y,
                landmarks[self.LEFT_HIP].z
            ]),
            'right_hip': np.array([
                landmarks[self.RIGHT_HIP].x,
                landmarks[self.RIGHT_HIP].y,
                landmarks[self.RIGHT_HIP].z
            ]),
            'left_knee': np.array([
                landmarks[self.LEFT_KNEE].x,
                landmarks[self.LEFT_KNEE].y,
                landmarks[self.LEFT_KNEE].z
            ]),
            'right_knee': np.array([
                landmarks[self.RIGHT_KNEE].x,
                landmarks[self.RIGHT_KNEE].y,
                landmarks[self.RIGHT_KNEE].z
            ]),
            'left_ankle': np.array([
                landmarks[self.LEFT_ANKLE].x,
                landmarks[self.LEFT_ANKLE].y,
                landmarks[self.LEFT_ANKLE].z
            ]),
            'right_ankle': np.array([
                landmarks[self.RIGHT_ANKLE].x,
                landmarks[self.RIGHT_ANKLE].y,
                landmarks[self.RIGHT_ANKLE].z
            ])
        }

    def _calculate_features_from_landmarks(self, world_landmarks) -> Optional[np.ndarray]:
        """
        Calculate motion features from pose landmarks.

        Returns:
            Feature array or None if calculation fails
        """
        try:
            coords = self._extract_landmark_coordinates(world_landmarks)
            
            motion_features = self.features_service.extract_all_features(
                left_shoulder=coords['left_shoulder'],
                right_shoulder=coords['right_shoulder'],
                left_hip=coords['left_hip'],
                right_hip=coords['right_hip'],
                left_knee=coords['left_knee'],
                right_knee=coords['right_knee'],
                left_ankle=coords['left_ankle'],
                right_ankle=coords['right_ankle']
            )
            
            # Convert to feature array in correct order
            features = np.array([
                motion_features.normalized_leg_length,
                motion_features.shoulder_vector_x,
                motion_features.shoulder_vector_z,
                motion_features.ankle_vector_x,
                motion_features.ankle_vector_z,
                motion_features.average_hip_angle,
                motion_features.average_knee_angle
            ])
            
            return features
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None

    def _predict_activity(self, features: np.ndarray) -> tuple:
        """
        Predict activity from features.

        Returns:
            Tuple of (predicted_label, confidence, all_probabilities)
        """
        # Normalize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get label
        predicted_label = self.index_to_label[prediction]
        confidence = probabilities[prediction]
        
        return predicted_label, confidence, probabilities

    def _draw_statistics(
        self,
        frame: np.ndarray,
        current_prediction: str,
        current_confidence: float,
        all_probabilities: np.ndarray,
        fps: float
    ):
        """Draw statistics on frame."""
        height, width = frame.shape[:2]
        
        # Background panel for statistics
        panel_height = 300
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Current prediction (large)
        font_large = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_large = 1.2
        thickness_large = 3
        
        text = f"{current_prediction}"
        text_size = cv2.getTextSize(text, font_large, font_scale_large, thickness_large)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50
        
        # Color based on confidence
        if current_confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif current_confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(
            panel, text, (text_x, text_y),
            font_large, font_scale_large, color, thickness_large
        )
        
        # Confidence
        conf_text = f"Confidence: {current_confidence*100:.1f}%"
        cv2.putText(
            panel, conf_text, (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            panel, fps_text, (width - 150, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Top 3 predictions
        top_indices = np.argsort(all_probabilities)[-3:][::-1]
        y_pos = 130
        cv2.putText(
            panel, "Top Predictions:", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )
        
        for i, idx in enumerate(top_indices):
            label = self.index_to_label[idx]
            prob = all_probabilities[idx]
            y_pos += 30
            text = f"  {i+1}. {label}: {prob*100:.1f}%"
            cv2.putText(
                panel, text, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Recent activity distribution
        if len(self.prediction_history) > 0:
            counter = Counter(self.prediction_history)
            y_pos = 130
            cv2.putText(
                panel, "Recent Activity:", (width // 2, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
            )
            
            for i, (label, count) in enumerate(counter.most_common(3)):
                percentage = (count / len(self.prediction_history)) * 100
                y_pos += 30
                text = f"  {label}: {percentage:.0f}%"
                cv2.putText(
                    panel, text, (width // 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        
        # Overlay panel on frame
        frame[height - panel_height:, :] = cv2.addWeighted(
            frame[height - panel_height:, :], 0.3, panel, 0.7, 0
        )

    def _process_frame(
        self,
        frame: np.ndarray,
        pose_detector
    ) -> tuple:
        """
        Process a single frame and return predictions.

        Returns:
            Tuple of (predicted_label, confidence, all_probabilities, has_pose)
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = pose_detector.process(frame_rgb)
        
        # Draw pose landmarks
        has_pose = results.pose_landmarks is not None
        
        if has_pose:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Extract features and predict
            if results.pose_world_landmarks:
                features = self._calculate_features_from_landmarks(
                    results.pose_world_landmarks.landmark
                )
                
                if features is not None:
                    predicted_label, confidence, all_probs = self._predict_activity(features)
                    
                    # Update history
                    self.prediction_history.append(predicted_label)
                    self.confidence_history.append(confidence)
                    
                    return predicted_label, confidence, all_probs, True
        
        # No pose detected
        predicted_label = "No pose detected"
        confidence = 0.0
        all_probs = np.zeros(len(self.index_to_label))
        return predicted_label, confidence, all_probs, False

    def run(self):
        """Run real-time classification."""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize MediaPipe Pose
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_detector:
            
            fps_counter = 0
            fps_time = cv2.getTickCount()
            fps = 0.0
            
            print("Starting real-time classification...")
            print("Make sure you're visible to the camera!")
            print()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                predicted_label, confidence, all_probs, has_pose = self._process_frame(
                    frame, pose_detector
                )
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 10 == 0:
                    current_time = cv2.getTickCount()
                    fps = 10.0 / ((current_time - fps_time) / cv2.getTickFrequency())
                    fps_time = current_time
                
                # Draw statistics
                if has_pose:
                    self._draw_statistics(
                        frame, predicted_label, confidence, all_probs, fps
                    )
                else:
                    # Show "No pose detected" message
                    height, width = frame.shape[:2]
                    text = "No pose detected - Please stand in front of camera"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = height - 50
                    cv2.putText(
                        frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )
                
                # Display frame
                cv2.imshow('Real-time Motion Classifier', frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nClassification stopped.")


def main():
    """Main entry point."""
    import sys
    
    TRAIN_MESSAGE = "Please train the model first using: python example_train_xgboost.py"
    
    # Check if model files exist
    model_path = Path("output/xgboost/xgboost_classifier.pkl")
    scaler_path = Path("output/xgboost/scaler.pkl")
    label_mappings_path = Path("output/xgboost/label_mappings.json")
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print(TRAIN_MESSAGE)
        sys.exit(1)
    
    if not scaler_path.exists():
        print(f"Error: Scaler file not found: {scaler_path}")
        print(TRAIN_MESSAGE)
        sys.exit(1)
    
    if not label_mappings_path.exists():
        print(f"Error: Label mappings file not found: {label_mappings_path}")
        print(TRAIN_MESSAGE)
        sys.exit(1)
    
    # Create and run classifier
    classifier = RealTimeMotionClassifier(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        label_mappings_path=str(label_mappings_path),
        camera_index=0
    )
    
    try:
        classifier.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

