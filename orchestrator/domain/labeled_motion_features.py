"""
Domain model for motion features with activity labels.
"""

from dataclasses import dataclass
from typing import Optional
from data_features.domain.motion_features import MotionFeatures
from labels_reader.domain.video_labels import ActivityLabel


@dataclass
class LabeledMotionFeatures:
    """
    Represents motion features for a frame along with its activity label.

    Attributes:
        motion_features: The calculated motion features for the frame
        activity_label: The activity label for this frame (if available)
        video_id: Identifier of the video
        frame_number: Frame number in the video
    """
    motion_features: MotionFeatures
    activity_label: Optional[ActivityLabel]
    video_id: str
    frame_number: int

    def has_label(self) -> bool:
        """Check if this frame has an associated activity label."""
        return self.activity_label is not None

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "video_id": self.video_id,
            "frame_number": self.frame_number,
            "activity_label": self.activity_label.value if self.activity_label else None,
            "motion_features": {
                "normalized_leg_length": self.motion_features.normalized_leg_length,
                "shoulder_vector_x": self.motion_features.shoulder_vector_x,
                "shoulder_vector_z": self.motion_features.shoulder_vector_z,
                "ankle_vector_x": self.motion_features.ankle_vector_x,
                "ankle_vector_z": self.motion_features.ankle_vector_z,
                "average_hip_angle": self.motion_features.average_hip_angle,
                "average_knee_angle": self.motion_features.average_knee_angle
            }
        }
