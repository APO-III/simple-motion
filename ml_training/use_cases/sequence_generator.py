"""
Sequence generator use case.

This module converts frame-by-frame motion features (from CSV) into
temporal sequences suitable for LSTM training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from ml_training.domain.sequence import (
    MotionSequence,
    SequenceDataset,
    SequenceGeneratorConfig
)


class SequenceGenerator:
    """
    Generates temporal sequences from motion features CSV files.

    This service converts individual frame features into sliding window
    sequences that capture temporal dynamics of activities.

    The process:
    1. Read CSV with frame-by-frame features
    2. Group frames by video_id
    3. Detect activity segments (continuous frames with same label)
    4. Generate sliding windows within each segment
    5. Extract feature matrices (window_size, num_features)
    """

    def __init__(self, config: Optional[SequenceGeneratorConfig] = None):
        """
        Initialize sequence generator.

        Args:
            config: Configuration for sequence generation.
                   If None, uses default configuration.
        """
        self.config = config or SequenceGeneratorConfig()

    def generate_from_csv(
        self,
        csv_path: str,
        source_name: Optional[str] = None,
        verbose: bool = True
    ) -> SequenceDataset:
        """
        Generate sequences from a CSV file.

        Args:
            csv_path: Path to CSV file with motion features
            source_name: Optional name for this data source (e.g., "source1")
            verbose: Print progress information

        Returns:
            SequenceDataset with generated sequences

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV has invalid format
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"GENERATING SEQUENCES FROM: {csv_path.name}")
            print(f"{'='*70}")
            print(f"Configuration:")
            print(f"  Window size: {self.config.window_size} frames")
            print(f"  Stride: {self.config.stride} frames")
            print(f"  Overlap: {((self.config.window_size - self.config.stride) / self.config.window_size) * 100:.1f}%")
            print(f"  Min segment length: {self.config.min_segment_length} frames")
            print()

        # Step 1: Read CSV
        df = pd.read_csv(csv_path)
        self._validate_csv(df)

        if verbose:
            print(f"✓ Loaded {len(df)} frames from CSV")

        # Step 2: Group by video
        videos = self._group_by_video(df)

        if verbose:
            print(f"✓ Found {len(videos)} videos")
            print()

        # Step 3: Generate sequences for each video
        all_sequences = []
        stats = {
            'videos_processed': 0,
            'segments_found': 0,
            'segments_skipped': 0,
            'sequences_generated': 0
        }

        for video_id, video_df in videos.items():
            if verbose:
                print(f"Processing video: {video_id}")

            # Detect activity segments
            segments = self._detect_segments(video_df)

            if verbose:
                print(f"  Found {len(segments)} segment(s)")

            for segment in segments:
                stats['segments_found'] += 1

                # Check minimum length
                if segment['num_frames'] < self.config.min_segment_length:
                    stats['segments_skipped'] += 1
                    if verbose:
                        print(f"  ⚠ Skipped segment [{segment['start_frame']}-{segment['end_frame']}] "
                              f"'{segment['label']}': too short ({segment['num_frames']} frames)")
                    continue

                # Generate sliding windows
                sequences = self._generate_sequences_for_segment(
                    video_df=video_df,
                    segment=segment,
                    video_id=video_id,
                    source_name=source_name
                )

                all_sequences.extend(sequences)
                stats['sequences_generated'] += len(sequences)

                if verbose:
                    print(f"  ✓ Segment [{segment['start_frame']}-{segment['end_frame']}] "
                          f"'{segment['label']}': {len(sequences)} sequences")

            stats['videos_processed'] += 1

        # Step 4: Build label mappings
        unique_labels = sorted(set(seq.label for seq in all_sequences))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}

        # Create dataset
        dataset = SequenceDataset(
            sequences=all_sequences,
            label_to_index=label_to_index,
            index_to_label=index_to_label
        )

        if verbose:
            print()
            print(f"{'='*70}")
            print(f"GENERATION SUMMARY")
            print(f"{'='*70}")
            print(f"Videos processed: {stats['videos_processed']}")
            print(f"Segments found: {stats['segments_found']}")
            print(f"Segments skipped (too short): {stats['segments_skipped']}")
            print(f"Sequences generated: {stats['sequences_generated']}")
            print(f"{'='*70}\n")

        return dataset

    def generate_from_multiple_csvs(
        self,
        csv_paths: List[Tuple[str, str]],
        verbose: bool = True
    ) -> SequenceDataset:
        """
        Generate sequences from multiple CSV files and merge them.

        Args:
            csv_paths: List of tuples (csv_path, source_name)
                      Example: [("data/source1.csv", "source1"),
                               ("data/source2.csv", "source2")]
            verbose: Print progress information

        Returns:
            Combined SequenceDataset with all sequences
        """
        all_sequences = []

        for csv_path, source_name in csv_paths:
            dataset = self.generate_from_csv(
                csv_path=csv_path,
                source_name=source_name,
                verbose=verbose
            )
            all_sequences.extend(dataset.sequences)

        # Build unified label mappings
        unique_labels = sorted(set(seq.label for seq in all_sequences))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}

        combined_dataset = SequenceDataset(
            sequences=all_sequences,
            label_to_index=label_to_index,
            index_to_label=index_to_label
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"COMBINED DATASET FROM {len(csv_paths)} SOURCES")
            print(f"{'='*70}")
            combined_dataset.print_statistics()

        return combined_dataset

    def _validate_csv(self, df: pd.DataFrame):
        """Validate that CSV has required columns."""
        required_columns = ['video_id', 'frame_number', 'activity_label'] + self.config.feature_columns

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")

    def _group_by_video(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group frames by video_id and sort by frame_number.

        Args:
            df: DataFrame with all frames

        Returns:
            Dictionary mapping video_id to DataFrame of frames
        """
        videos = {}
        for video_id, group in df.groupby('video_id'):
            # Sort by frame number to ensure temporal order
            videos[video_id] = group.sort_values('frame_number').reset_index(drop=True)

        return videos

    def _detect_segments(self, video_df: pd.DataFrame) -> List[Dict]:
        """
        Detect continuous segments with the same activity label.

        Args:
            video_df: DataFrame with frames from one video (sorted by frame_number)

        Returns:
            List of segment dictionaries with keys:
            - start_frame: First frame number
            - end_frame: Last frame number
            - label: Activity label
            - num_frames: Number of frames in segment
        """
        if len(video_df) == 0:
            return []

        segments = []
        current_label = video_df.iloc[0]['activity_label']
        start_frame = video_df.iloc[0]['frame_number']
        start_idx = 0

        for i in range(1, len(video_df)):
            row = video_df.iloc[i]

            # Check for label change
            if row['activity_label'] != current_label:
                # Save previous segment
                end_frame = video_df.iloc[i - 1]['frame_number']
                segments.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'label': current_label,
                    'num_frames': i - start_idx
                })

                # Start new segment
                current_label = row['activity_label']
                start_frame = row['frame_number']
                start_idx = i

        # Add last segment
        end_frame = video_df.iloc[-1]['frame_number']
        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_idx': start_idx,
            'end_idx': len(video_df) - 1,
            'label': current_label,
            'num_frames': len(video_df) - start_idx
        })

        return segments

    def _generate_sequences_for_segment(
        self,
        video_df: pd.DataFrame,
        segment: Dict,
        video_id: str,
        source_name: Optional[str]
    ) -> List[MotionSequence]:
        """
        Generate sliding window sequences for a single segment.

        Args:
            video_df: DataFrame with all frames from the video
            segment: Segment dictionary from _detect_segments
            video_id: Video identifier
            source_name: Data source name

        Returns:
            List of MotionSequence objects
        """
        sequences = []

        # Extract segment data
        segment_df = video_df.iloc[segment['start_idx']:segment['end_idx'] + 1]

        # Generate sliding windows
        window_start_idx = 0
        while window_start_idx + self.config.window_size <= len(segment_df):
            window_end_idx = window_start_idx + self.config.window_size

            # Extract window data
            window_df = segment_df.iloc[window_start_idx:window_end_idx]

            # Extract features matrix (window_size, num_features)
            features_matrix = window_df[self.config.feature_columns].values

            # Create MotionSequence
            sequence = MotionSequence(
                features=features_matrix,
                label=segment['label'],
                video_id=video_id,
                start_frame=int(window_df.iloc[0]['frame_number']),
                end_frame=int(window_df.iloc[-1]['frame_number']),
                source=source_name
            )

            sequences.append(sequence)

            # Move to next window
            window_start_idx += self.config.stride

        return sequences
