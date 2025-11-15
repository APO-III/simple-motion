"""
Domain models for motion sequences.

This module defines the core data structures for temporal motion sequences
used in LSTM training.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class MotionSequence:
    """
    Represents a temporal sequence of motion features.

    This is a window of consecutive frames that captures the temporal
    dynamics of a motion activity.

    Attributes:
        features: NumPy array of shape (timesteps, num_features)
                 Example: (30, 7) for 30 frames with 7 features each
        label: Activity label for this sequence
        video_id: Source video identifier
        start_frame: First frame number in this sequence
        end_frame: Last frame number in this sequence
        source: Data source identifier (e.g., "source1", "source2", "source3")
    """
    features: np.ndarray  # Shape: (timesteps, num_features)
    label: str
    video_id: str
    start_frame: int
    end_frame: int
    source: Optional[str] = None

    def __post_init__(self):
        """Validate sequence after initialization."""
        if not isinstance(self.features, np.ndarray):
            raise ValueError("features must be a NumPy array")

        if len(self.features.shape) != 2:
            raise ValueError(f"features must be 2D array, got shape {self.features.shape}")

        if self.start_frame > self.end_frame:
            raise ValueError(f"start_frame ({self.start_frame}) must be <= end_frame ({self.end_frame})")

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps (frames) in this sequence."""
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        """Number of features per timestep."""
        return self.features.shape[1]

    @property
    def shape(self) -> tuple:
        """Shape of the features array."""
        return self.features.shape

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "video_id": self.video_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "label": self.label,
            "source": self.source,
            "shape": self.shape,
            "num_timesteps": self.num_timesteps,
            "num_features": self.num_features
        }


@dataclass
class SequenceDataset:
    """
    Collection of motion sequences ready for ML training.

    Attributes:
        sequences: List of MotionSequence objects
        label_to_index: Mapping from label names to integer indices
        index_to_label: Mapping from integer indices to label names
    """
    sequences: List[MotionSequence]
    label_to_index: dict
    index_to_label: dict

    def __len__(self) -> int:
        """Number of sequences in the dataset."""
        return len(self.sequences)

    def get_X(self) -> np.ndarray:
        """
        Get feature arrays as a single NumPy array.

        Returns:
            NumPy array of shape (num_sequences, timesteps, num_features)
        """
        return np.array([seq.features for seq in self.sequences])

    def get_y(self) -> np.ndarray:
        """
        Get encoded labels as a NumPy array of integers.

        Returns:
            NumPy array of shape (num_sequences,) with integer class indices
        """
        return np.array([self.label_to_index[seq.label] for seq in self.sequences])

    def get_y_categorical(self, num_classes: int) -> np.ndarray:
        """
        Get one-hot encoded labels.

        Args:
            num_classes: Number of activity classes

        Returns:
            NumPy array of shape (num_sequences, num_classes)
        """
        from tensorflow.keras.utils import to_categorical
        return to_categorical(self.get_y(), num_classes=num_classes)

    def get_metadata(self) -> List[dict]:
        """Get metadata for all sequences."""
        return [seq.to_dict() for seq in self.sequences]

    def get_statistics(self) -> dict:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if not self.sequences:
            return {
                "num_sequences": 0,
                "num_classes": 0,
                "class_distribution": {},
                "videos": [],
                "sources": [],
                "num_videos": 0,
                "num_sources": 0,
                "sequence_shape": None
            }

        # Count labels
        label_counts = {}
        videos = set()
        sources = set()

        for seq in self.sequences:
            label_counts[seq.label] = label_counts.get(seq.label, 0) + 1
            videos.add(seq.video_id)
            if seq.source:
                sources.add(seq.source)

        return {
            "num_sequences": len(self.sequences),
            "num_classes": len(label_counts),
            "class_distribution": label_counts,
            "videos": sorted(list(videos)),
            "sources": sorted(list(sources)),
            "num_videos": len(videos),
            "num_sources": len(sources),
            "sequence_shape": self.sequences[0].shape if self.sequences else None
        }

    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("SEQUENCE DATASET STATISTICS")
        print("=" * 70)
        print(f"Total sequences: {stats['num_sequences']}")
        print(f"Sequence shape: {stats['sequence_shape']}")
        print(f"Number of videos: {stats['num_videos']}")
        print(f"Number of sources: {stats['num_sources']}")
        print(f"Number of classes: {stats['num_classes']}")
        print()

        if stats['class_distribution']:
            print("Class distribution:")
            total = stats['num_sequences']
            for label in sorted(stats['class_distribution'].keys()):
                count = stats['class_distribution'][label]
                percentage = (count / total) * 100
                print(f"  {label:30s}: {count:5d} sequences ({percentage:5.2f}%)")

        if stats['sources']:
            print(f"\nSources: {', '.join(stats['sources'])}")

        print("=" * 70 + "\n")


@dataclass
class SequenceGeneratorConfig:
    """
    Configuration for sequence generation.

    Attributes:
        window_size: Number of consecutive frames per sequence
        stride: Number of frames to skip between windows (controls overlap)
        min_segment_length: Minimum frames required for a segment to be processed
        feature_columns: List of feature column names from CSV
    """
    window_size: int = 30
    stride: int = 15
    min_segment_length: Optional[int] = None
    feature_columns: List[str] = None

    def __post_init__(self):
        """Set defaults and validate."""
        if self.min_segment_length is None:
            self.min_segment_length = self.window_size

        if self.feature_columns is None:
            self.feature_columns = [
                "normalized_leg_length",
                "shoulder_vector_x",
                "shoulder_vector_z",
                "ankle_vector_x",
                "ankle_vector_z",
                "average_hip_angle",
                "average_knee_angle"
            ]

        # Validate
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

        if self.stride <= 0:
            raise ValueError("stride must be positive")

        if self.min_segment_length < self.window_size:
            raise ValueError("min_segment_length must be >= window_size")

    @property
    def num_features(self) -> int:
        """Number of features per frame."""
        return len(self.feature_columns)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "stride": self.stride,
            "min_segment_length": self.min_segment_length,
            "num_features": self.num_features,
            "overlap_percentage": ((self.window_size - self.stride) / self.window_size) * 100
        }
