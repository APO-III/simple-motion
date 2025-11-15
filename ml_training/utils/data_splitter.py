"""
Data splitting utilities for train/validation/test sets.

This module provides stratified splitting that ensures:
- Videos are not split across train/val/test sets
- Class distribution is balanced across splits
- Source distribution is balanced (when splitting multiple sources)
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import defaultdict
import random

from ml_training.domain.sequence import SequenceDataset, MotionSequence


class DataSplitter:
    """
    Splits sequence datasets into train/validation/test sets.

    Key principle: Split at VIDEO level, not sequence level.
    This prevents data leakage where sequences from the same video
    appear in both training and test sets.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize data splitter.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def split_by_video(
        self,
        dataset: SequenceDataset,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by_label: bool = True,
        verbose: bool = True
    ) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
        """
        Split dataset into train/val/test sets at the VIDEO level.

        Args:
            dataset: Input dataset to split
            train_ratio: Proportion for training set (default: 0.70)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for test set (default: 0.15)
            stratify_by_label: Try to balance class distribution across splits
            verbose: Print split statistics

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        # Group sequences by video
        video_to_sequences = defaultdict(list)
        for seq in dataset.sequences:
            video_to_sequences[seq.video_id].append(seq)

        video_ids = list(video_to_sequences.keys())

        if verbose:
            print(f"\n{'='*70}")
            print("SPLITTING DATASET BY VIDEO")
            print(f"{'='*70}")
            print(f"Total videos: {len(video_ids)}")
            print(f"Total sequences: {len(dataset.sequences)}")
            print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
            print()

        # Split videos
        if stratify_by_label:
            train_videos, val_videos, test_videos = self._stratified_split_videos(
                video_to_sequences=video_to_sequences,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
        else:
            train_videos, val_videos, test_videos = self._random_split_videos(
                video_ids=video_ids,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )

        # Create sequence lists for each split
        train_sequences = []
        val_sequences = []
        test_sequences = []

        for video_id in train_videos:
            train_sequences.extend(video_to_sequences[video_id])

        for video_id in val_videos:
            val_sequences.extend(video_to_sequences[video_id])

        for video_id in test_videos:
            test_sequences.extend(video_to_sequences[video_id])

        # Create datasets
        train_dataset = SequenceDataset(
            sequences=train_sequences,
            label_to_index=dataset.label_to_index,
            index_to_label=dataset.index_to_label
        )

        val_dataset = SequenceDataset(
            sequences=val_sequences,
            label_to_index=dataset.label_to_index,
            index_to_label=dataset.index_to_label
        )

        test_dataset = SequenceDataset(
            sequences=test_sequences,
            label_to_index=dataset.label_to_index,
            index_to_label=dataset.index_to_label
        )

        if verbose:
            self._print_split_statistics(train_dataset, val_dataset, test_dataset)

        return train_dataset, val_dataset, test_dataset

    def _random_split_videos(
        self,
        video_ids: List[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Randomly split video IDs.

        Args:
            video_ids: List of video identifiers
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion

        Returns:
            Tuple of (train_video_ids, val_video_ids, test_video_ids)
        """
        # Shuffle videos
        shuffled = video_ids.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        num_videos = len(shuffled)
        train_end = int(num_videos * train_ratio)
        val_end = train_end + int(num_videos * val_ratio)

        train_videos = shuffled[:train_end]
        val_videos = shuffled[train_end:val_end]
        test_videos = shuffled[val_end:]

        return train_videos, val_videos, test_videos

    def _stratified_split_videos(
        self,
        video_to_sequences: dict,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split videos while trying to balance class distribution.

        This is a best-effort stratification. Since we split at video level
        and videos may have different activity distributions, perfect
        stratification may not be possible.

        Args:
            video_to_sequences: Mapping from video_id to list of sequences
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion

        Returns:
            Tuple of (train_video_ids, val_video_ids, test_video_ids)
        """
        # Get dominant label for each video
        video_labels = {}
        for video_id, sequences in video_to_sequences.items():
            # Find most common label in this video
            label_counts = defaultdict(int)
            for seq in sequences:
                label_counts[seq.label] += 1
            dominant_label = max(label_counts.items(), key=lambda x: x[1])[0]
            video_labels[video_id] = dominant_label

        # Group videos by dominant label
        label_to_videos = defaultdict(list)
        for video_id, label in video_labels.items():
            label_to_videos[label].append(video_id)

        # Split each label group
        train_videos = []
        val_videos = []
        test_videos = []

        for label, videos in label_to_videos.items():
            # Shuffle videos for this label
            shuffled = videos.copy()
            random.shuffle(shuffled)

            # Calculate splits
            num_videos = len(shuffled)
            train_end = max(1, int(num_videos * train_ratio))
            val_end = train_end + max(0, int(num_videos * val_ratio))

            train_videos.extend(shuffled[:train_end])
            val_videos.extend(shuffled[train_end:val_end])
            test_videos.extend(shuffled[val_end:])

        return train_videos, val_videos, test_videos

    def _print_split_statistics(
        self,
        train_dataset: SequenceDataset,
        val_dataset: SequenceDataset,
        test_dataset: SequenceDataset
    ):
        """Print statistics for each split."""
        print(f"SPLIT RESULTS:")
        print(f"{'-'*70}")

        splits = [
            ("TRAIN", train_dataset),
            ("VALIDATION", val_dataset),
            ("TEST", test_dataset)
        ]

        for split_name, dataset in splits:
            stats = dataset.get_statistics()
            total_sequences = stats['num_sequences']

            print(f"\n{split_name} SET:")
            print(f"  Videos: {stats['num_videos']}")
            print(f"  Sequences: {total_sequences}")

            if stats['class_distribution']:
                print(f"  Class distribution:")
                for label in sorted(stats['class_distribution'].keys()):
                    count = stats['class_distribution'][label]
                    percentage = (count / total_sequences) * 100 if total_sequences > 0 else 0
                    print(f"    {label:30s}: {count:4d} ({percentage:5.2f}%)")

        print(f"\n{'='*70}\n")
