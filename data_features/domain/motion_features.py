from dataclasses import dataclass
from typing import Optional


@dataclass
class MotionFeatures:
    """
    Domain model containing all motion analysis features.
    """
    normalized_leg_length: float
    shoulder_vector_x: float
    shoulder_vector_z: float
    ankle_vector_x: float
    ankle_vector_z: float
    average_hip_angle: float
    average_knee_angle: float

    def __post_init__(self):
        """Validate feature values after initialization."""
        self._validate_normalized_values()
        self._validate_angles()

    def _validate_normalized_values(self):
        """Ensure normalized distances are non-negative."""
        if self.normalized_leg_length < 0:
            raise ValueError("normalized_leg_length must be non-negative")
        

    def _validate_angles(self):
        """Ensure angles are within valid range [0, 180]."""
        if not 0 <= self.average_hip_angle <= 180:
            raise ValueError("average_hip_angle must be between 0 and 180 degrees")
        if not 0 <= self.average_knee_angle <= 180:
            raise ValueError("average_knee_angle must be between 0 and 180 degrees")

    def to_dict(self) -> dict:
        """Convert features to dictionary format."""
        return {
            "normalized_leg_length": self.normalized_leg_length,
            "shoulder_vector_x": self.shoulder_vector_x,
            "shoulder_vector_z": self.shoulder_vector_z,
            "ankle_vector_x": self.ankle_vector_x,
            "ankle_vector_z": self.ankle_vector_z,
            "average_hip_angle": self.average_hip_angle,
            "average_knee_angle": self.average_knee_angle
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MotionFeatures':
        """Create MotionFeatures instance from dictionary."""
        return cls(
            normalized_leg_length=data["normalized_leg_length"],
            shoulder_vector_x=data["shoulder_vector_x"],
            shoulder_vector_z=data["shoulder_vector_z"],
            ankle_vector_x=data["ankle_vector_x"],
            ankle_vector_z=data["ankle_vector_z"],
            average_hip_angle=data["average_hip_angle"],
            average_knee_angle=data["average_knee_angle"]
        )
