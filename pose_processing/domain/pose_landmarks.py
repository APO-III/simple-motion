from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LandmarkPoint:
    """
    Single landmark point with coordinates.
    """
    x: float
    y: float
    z: float
    visibility: Optional[float] = None

    @classmethod
    def from_mediapipe_landmark(cls, landmark) -> 'LandmarkPoint':
        """Create LandmarkPoint from MediaPipe landmark."""
        return cls(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=getattr(landmark, 'visibility', None)
        )


@dataclass
class PoseLandmarks:
    """
    Domain model containing pose landmarks from MediaPipe.
    """
    video_id: str
    frame_number: int
    timestamp: float
    landmarks: List[LandmarkPoint]
    world_landmarks: List[LandmarkPoint]

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "video_id": self.video_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in self.landmarks
            ],
            "world_landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in self.world_landmarks
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PoseLandmarks':
        """Create PoseLandmarks from dictionary."""
        return cls(
            video_id=data["video_id"],
            frame_number=data["frame_number"],
            timestamp=data["timestamp"],
            landmarks=[
                LandmarkPoint(
                    x=lm["x"],
                    y=lm["y"],
                    z=lm["z"],
                    visibility=lm.get("visibility")
                )
                for lm in data["landmarks"]
            ],
            world_landmarks=[
                LandmarkPoint(
                    x=lm["x"],
                    y=lm["y"],
                    z=lm["z"],
                    visibility=lm.get("visibility")
                )
                for lm in data["world_landmarks"]
            ]
        )
