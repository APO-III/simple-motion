from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ActivityLabel(Enum):
    """Enumeration of possible activity labels."""
    SITTING_DOWN = "sitting_down"
    STANDING_UP = "standing_up"
    WALKING_TOWARDS_CAMERA = "walking_towards_camera"
    WALKING_AWAY_FROM_CAMERA = "walking_away_from_camera"
    TURNING = "turning"
    STANDING_STILL = "standing_still"
    SITTING_STILL = "sitting_still"

    @classmethod
    def from_string(cls, label: str) -> 'ActivityLabel':
        """Create ActivityLabel from string."""
        for activity in cls:
            if activity.value == label:
                return activity
        raise ValueError(f"Unknown activity label: {label}")


@dataclass
class FrameRange:
    """
    Represents a range of frames with an activity label.
    """
    start_frame: int
    end_frame: int
    activity: ActivityLabel

    def contains_frame(self, frame_number: int) -> bool:
        """Check if frame number is within this range."""
        return self.start_frame <= frame_number <= self.end_frame

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "activity": self.activity.value
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FrameRange':
        """Create FrameRange from dictionary."""
        return cls(
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            activity=ActivityLabel.from_string(data["activity"])
        )


@dataclass
class VideoAnnotation:
    """
    Contains all annotations for a single video.
    """
    video_id: str
    frame_ranges: List[FrameRange] = field(default_factory=list)

    def get_label_for_frame(self, frame_number: int) -> Optional[ActivityLabel]:
        """
        Get activity label for a specific frame.

        Args:
            frame_number: Frame number to query

        Returns:
            ActivityLabel if frame is labeled, None otherwise
        """
        for frame_range in self.frame_ranges:
            if frame_range.contains_frame(frame_number):
                return frame_range.activity
        return None

    def add_frame_range(self, start: int, end: int, activity: ActivityLabel):
        """Add a new frame range annotation."""
        self.frame_ranges.append(FrameRange(start, end, activity))

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "video_id": self.video_id,
            "frame_ranges": [fr.to_dict() for fr in self.frame_ranges]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VideoAnnotation':
        """Create VideoAnnotation from dictionary."""
        return cls(
            video_id=data["video_id"],
            frame_ranges=[FrameRange.from_dict(fr) for fr in data["frame_ranges"]]
        )


@dataclass
class LabelsDataset:
    """
    Contains annotations for multiple videos.
    """
    annotations: Dict[str, VideoAnnotation] = field(default_factory=dict)

    def add_video_annotation(self, video_annotation: VideoAnnotation):
        """Add video annotation to dataset."""
        self.annotations[video_annotation.video_id] = video_annotation

    def get_video_annotation(self, video_id: str) -> Optional[VideoAnnotation]:
        """Get annotation for specific video."""
        return self.annotations.get(video_id)

    def get_all_video_ids(self) -> List[str]:
        """Get list of all video IDs in dataset."""
        return list(self.annotations.keys())

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "annotations": {
                video_id: annotation.to_dict()
                for video_id, annotation in self.annotations.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LabelsDataset':
        """Create LabelsDataset from dictionary."""
        dataset = cls()
        for video_id, annotation_data in data["annotations"].items():
            dataset.add_video_annotation(VideoAnnotation.from_dict(annotation_data))
        return dataset
