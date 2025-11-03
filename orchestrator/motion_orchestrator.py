import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path
from data_features.use_cases.features_service import FeaturesService
from mediapipe.use_cases.pose_extraction_service import PoseExtractionService
from mediapipe.domain.pose_landmarks import PoseLandmarks
from data_features.domain.motion_features import MotionFeatures


class MotionOrchestrator:
    """
    Orchestrator that coordinates pose extraction and feature calculation.
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
        pose_extraction_service: PoseExtractionService,
        features_service: FeaturesService
    ):
        """
        Initialize MotionOrchestrator.

        Args:
            pose_extraction_service: Service for extracting pose landmarks
            features_service: Service for calculating motion features
        """
        self.pose_extraction_service = pose_extraction_service
        self.features_service = features_service

    def _extract_landmark_coordinates(
        self,
        pose_landmarks: PoseLandmarks,
        use_world: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract specific landmark coordinates as numpy arrays.

        Args:
            pose_landmarks: PoseLandmarks object
            use_world: Use world_landmarks (True) or standard landmarks (False)

        Returns:
            Dictionary with landmark names as keys and 3D coordinates as values
        """
        landmarks = pose_landmarks.world_landmarks if use_world else pose_landmarks.landmarks

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

    def _calculate_features_from_landmarks(
        self,
        pose_landmarks: PoseLandmarks
    ) -> MotionFeatures:
        """
        Calculate motion features from pose landmarks.

        Args:
            pose_landmarks: PoseLandmarks object

        Returns:
            MotionFeatures object
        """
        coords = self._extract_landmark_coordinates(pose_landmarks, use_world=True)

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

        return motion_features

    def _generate_visualization(
        self,
        frame_number: int,
        motion_features: MotionFeatures,
        output_dir: str = "output_validations"
    ):
        """
        Generate visualization image with feature labels.

        Args:
            frame_number: Frame number
            motion_features: MotionFeatures object to visualize
            output_dir: Output directory for images
        """
        img_height = 800
        img_width = 600
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)
        line_height = 35
        start_y = 50

        title = f"Frame: {frame_number}"
        cv2.putText(img, title, (20, start_y), font, 1.0, (0, 0, 255), 2)

        features_text = [
            f"Normalized Leg Length: {motion_features.normalized_leg_length:.4f}",
            f"Normalized Shoulder Dist: {motion_features.normalized_shoulder_distance:.4f}",
            f"Shoulder Vector X: {motion_features.shoulder_vector_x:.4f}",
            f"Shoulder Vector Y: {motion_features.shoulder_vector_y:.4f}",
            f"Shoulder Vector Z: {motion_features.shoulder_vector_z:.4f}",
            f"Ankle Vector X: {motion_features.ankle_vector_x:.4f}",
            f"Ankle Vector Y: {motion_features.ankle_vector_y:.4f}",
            f"Ankle Vector Z: {motion_features.ankle_vector_z:.4f}",
            f"Average Hip Angle: {motion_features.average_hip_angle:.2f} deg",
            f"Average Knee Angle: {motion_features.average_knee_angle:.2f} deg"
        ]

        y_position = start_y + 60
        for text in features_text:
            cv2.putText(img, text, (20, y_position), font, font_scale, font_color, 1)
            y_position += line_height

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / f"frame_{frame_number:05d}.png"
        cv2.imwrite(str(filename), img)

    def process_video(
        self,
        video_path: str,
        target_fps: float = 10.0,
        video_id: str = None,
        generate_visualizations: bool = False,
        output_dir: str = "output_validations"
    ) -> Dict[int, MotionFeatures]:
        """
        Process video to extract motion features.

        Args:
            video_path: Path to video file
            target_fps: Sampling rate for frame extraction
            video_id: Optional video identifier
            generate_visualizations: Generate visualization images
            output_dir: Directory for visualization output

        Returns:
            Dictionary mapping frame numbers to MotionFeatures
        """
        pose_landmarks_list = self.pose_extraction_service.extract_landmarks_from_video(
            video_path=video_path,
            target_fps=target_fps,
            video_id=video_id
        )

        features_map = {}

        for pose_landmarks in pose_landmarks_list:
            motion_features = self._calculate_features_from_landmarks(pose_landmarks)
            frame_number = pose_landmarks.frame_number
            features_map[frame_number] = motion_features

            if generate_visualizations:
                self._generate_visualization(frame_number, motion_features, output_dir)

        return features_map
