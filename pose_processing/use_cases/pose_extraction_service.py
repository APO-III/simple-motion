import cv2
import mediapipe as mp
from typing import List, Optional, Tuple
from pose_processing.domain.pose_landmarks import PoseLandmarks, LandmarkPoint
import numpy as np


class PoseExtractionService:
    """
    Service for extracting pose landmarks from video using MediaPipe.
    """

    def __init__(self, mediapipe_style: int = 1):
        """
        Initialize PoseExtractionService.

        Args:
            mediapipe_style: Model complexity (0, 1, or 2)
                0 - Lite model (fastest, least accurate)
                1 - Full model (balanced)
                2 - Heavy model (slowest, most accurate)
        """
        self.mediapipe_style = mediapipe_style
        self.mp_pose = mp.solutions.pose

    def extract_landmarks_from_frame(
        self,
        frame: np.ndarray,
        pose_detector
    ) -> Optional[Tuple[List[LandmarkPoint], List[LandmarkPoint]]]:
        """
        Extract landmarks from a single frame.

        Args:
            frame: Input frame (BGR format)
            pose_detector: MediaPipe Pose detector instance

        Returns:
            Tuple of (landmarks, world_landmarks) or None if no pose detected
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(frame_rgb)

        if not results.pose_landmarks or not results.pose_world_landmarks:
            return None

        landmarks = [
            LandmarkPoint.from_mediapipe_landmark(lm)
            for lm in results.pose_landmarks.landmark
        ]

        world_landmarks = [
            LandmarkPoint.from_mediapipe_landmark(lm)
            for lm in results.pose_world_landmarks.landmark
        ]

        return landmarks, world_landmarks

    def extract_landmarks_from_video(
        self,
        video_path: str,
        target_fps: float,
        video_id: Optional[str] = None
    ) -> List[PoseLandmarks]:
        """
        Extract pose landmarks from video at specified FPS.

        Args:
            video_path: Path to the video file
            target_fps: Desired sampling rate (frames per second)
            video_id: Identifier for the video (defaults to video filename)

        Returns:
            List of PoseLandmarks for each sampled frame
        """
        if video_id is None:
            video_id = video_path.split('/')[-1].split('.')[0]

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / target_fps))

        pose_landmarks_list = []
        frame_count = 0
        sampled_frame_count = 0

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.mediapipe_style,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_detector:

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    result = self.extract_landmarks_from_frame(frame, pose_detector)

                    if result is not None:
                        landmarks, world_landmarks = result
                        timestamp = frame_count / original_fps

                        pose_landmarks = PoseLandmarks(
                            video_id=video_id,
                            frame_number=sampled_frame_count,
                            timestamp=timestamp,
                            landmarks=landmarks,
                            world_landmarks=world_landmarks
                        )

                        pose_landmarks_list.append(pose_landmarks)
                        sampled_frame_count += 1

                frame_count += 1

        cap.release()

        return pose_landmarks_list
