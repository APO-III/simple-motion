import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path
from data_features.use_cases.features_service import FeaturesService
from pose_processing.use_cases.pose_extraction_service import PoseExtractionService
from pose_processing.domain.pose_landmarks import PoseLandmarks
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
        frame: np.ndarray,
        frame_number: int,
        pose_landmarks: PoseLandmarks,
        motion_features: MotionFeatures,
        output_dir: str = "output_validations"
    ):
        """
        Generate visualization image with landmarks and feature labels.

        Args:
            frame: Original video frame
            frame_number: Frame number
            pose_landmarks: PoseLandmarks with landmark coordinates
            motion_features: MotionFeatures object to visualize
            output_dir: Output directory for images
        """
        import mediapipe as mp

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        img = frame.copy()
        img_height, img_width = img.shape[:2]

        # Draw landmarks on the frame
        for idx, landmark in enumerate(pose_landmarks.landmarks):
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        # Draw connections between landmarks
        connections = [
            (self.LEFT_SHOULDER, self.LEFT_HIP),
            (self.RIGHT_SHOULDER, self.RIGHT_HIP),
            (self.LEFT_HIP, self.LEFT_KNEE),
            (self.RIGHT_HIP, self.RIGHT_KNEE),
            (self.LEFT_KNEE, self.LEFT_ANKLE),
            (self.RIGHT_KNEE, self.RIGHT_ANKLE),
            (self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
            (self.LEFT_HIP, self.RIGHT_HIP)
        ]

        for connection in connections:
            start_idx, end_idx = connection
            start = pose_landmarks.landmarks[start_idx]
            end = pose_landmarks.landmarks[end_idx]

            start_point = (int(start.x * img_width), int(start.y * img_height))
            end_point = (int(end.x * img_width), int(end.y * img_height))

            cv2.line(img, start_point, end_point, (255, 0, 0), 2)

        # Add feature text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        line_height = 25
        start_y = 30

        features_text = [
            f"Frame: {frame_number}",
            f"Leg Length: {motion_features.normalized_leg_length:.3f}",
            f"Shoulder Vec X: {motion_features.shoulder_vector_x:.3f}",
            f"Shoulder Vec Z: {motion_features.shoulder_vector_z:.3f}",
            f"Ankle Vec X: {motion_features.ankle_vector_x:.3f}",
            f"Ankle Vec Z: {motion_features.ankle_vector_z:.3f}",
            f"Hip Angle: {motion_features.average_hip_angle:.1f}deg",
            f"Knee Angle: {motion_features.average_knee_angle:.1f}deg"
        ]

        y_position = start_y
        for text in features_text:
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            cv2.rectangle(img, (5, y_position - 20),
                         (text_size[0] + 15, y_position + 5), bg_color, -1)
            cv2.putText(img, text, (10, y_position), font, font_scale, font_color, 1)
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
        # Clear output directory if generating visualizations
        if generate_visualizations:
            import shutil
            output_path = Path(output_dir)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        # Extract pose landmarks from video
        pose_landmarks_list = self.pose_extraction_service.extract_landmarks_from_video(
            video_path=video_path,
            target_fps=target_fps,
            video_id=video_id
        )

        features_map = {}

        # If visualizations needed, read video frames
        if generate_visualizations:
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(original_fps / target_fps))

            video_frame_count = 0
            sampled_frame_count = 0
            frames_dict = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if video_frame_count % frame_interval == 0:
                    frames_dict[sampled_frame_count] = frame
                    sampled_frame_count += 1

                video_frame_count += 1

            cap.release()

            # Process landmarks and generate visualizations
            for pose_landmarks in pose_landmarks_list:
                motion_features = self._calculate_features_from_landmarks(pose_landmarks)
                frame_number = pose_landmarks.frame_number
                features_map[frame_number] = motion_features

                if frame_number in frames_dict:
                    self._generate_visualization(
                        frame=frames_dict[frame_number],
                        frame_number=frame_number,
                        pose_landmarks=pose_landmarks,
                        motion_features=motion_features,
                        output_dir=output_dir
                    )
        else:
            # Process without visualizations
            for pose_landmarks in pose_landmarks_list:
                motion_features = self._calculate_features_from_landmarks(pose_landmarks)
                frame_number = pose_landmarks.frame_number
                features_map[frame_number] = motion_features

        return features_map
