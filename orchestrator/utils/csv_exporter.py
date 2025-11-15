"""
CSV exporter utility for LabeledMotionFeatures.

This module provides functionality to export LabeledMotionFeatures to CSV format
with flattened motion features columns.
"""

import csv
from typing import List
from pathlib import Path
from orchestrator.domain.labeled_motion_features import LabeledMotionFeatures


class CSVExporter:
    """
    Exports LabeledMotionFeatures to CSV format with flattened columns.
    """

    # CSV column headers
    HEADERS = [
        'video_id',
        'frame_number',
        'activity_label',
        'normalized_leg_length',
        'shoulder_vector_x',
        'shoulder_vector_z',
        'ankle_vector_x',
        'ankle_vector_z',
        'average_hip_angle',
        'average_knee_angle'
    ]

    def export_to_csv(
        self,
        labeled_features_list: List[LabeledMotionFeatures],
        output_path: str,
        include_unlabeled: bool = True
    ) -> None:
        """
        Export LabeledMotionFeatures to CSV file.

        Args:
            labeled_features_list: List of LabeledMotionFeatures to export
            output_path: Path to output CSV file
            include_unlabeled: Include frames without activity labels (default: True)

        Raises:
            ValueError: If labeled_features_list is empty
        """
        if not labeled_features_list:
            raise ValueError("Cannot export empty labeled features list")

        # Filter data if needed
        data_to_export = labeled_features_list
        if not include_unlabeled:
            data_to_export = [lf for lf in labeled_features_list if lf.has_label()]

        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.HEADERS)
            writer.writeheader()

            for labeled_feature in data_to_export:
                row = self._flatten_labeled_feature(labeled_feature)
                writer.writerow(row)

        print(f"✓ Exported {len(data_to_export)} records to {output_path}")

    def _flatten_labeled_feature(self, labeled_feature: LabeledMotionFeatures) -> dict:
        """
        Flatten LabeledMotionFeatures into a single-level dictionary.

        Args:
            labeled_feature: LabeledMotionFeatures object to flatten

        Returns:
            Dictionary with flattened values ready for CSV export
        """
        motion_features = labeled_feature.motion_features

        return {
            'video_id': labeled_feature.video_id,
            'frame_number': labeled_feature.frame_number,
            'activity_label': labeled_feature.activity_label.value if labeled_feature.has_label() else '',
            'normalized_leg_length': motion_features.normalized_leg_length,
            'shoulder_vector_x': motion_features.shoulder_vector_x,
            'shoulder_vector_z': motion_features.shoulder_vector_z,
            'ankle_vector_x': motion_features.ankle_vector_x,
            'ankle_vector_z': motion_features.ankle_vector_z,
            'average_hip_angle': motion_features.average_hip_angle,
            'average_knee_angle': motion_features.average_knee_angle
        }

    def export_by_video(
        self,
        labeled_features_list: List[LabeledMotionFeatures],
        output_dir: str,
        include_unlabeled: bool = True
    ) -> None:
        """
        Export LabeledMotionFeatures to separate CSV files per video.

        Args:
            labeled_features_list: List of LabeledMotionFeatures to export
            output_dir: Directory for output CSV files
            include_unlabeled: Include frames without activity labels (default: True)

        Raises:
            ValueError: If labeled_features_list is empty
        """
        if not labeled_features_list:
            raise ValueError("Cannot export empty labeled features list")

        # Group by video_id
        videos_data = {}
        for labeled_feature in labeled_features_list:
            video_id = labeled_feature.video_id
            if video_id not in videos_data:
                videos_data[video_id] = []
            videos_data[video_id].append(labeled_feature)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export each video to separate CSV
        for video_id, features_list in videos_data.items():
            csv_filename = output_path / f"{video_id}.csv"
            self.export_to_csv(
                labeled_features_list=features_list,
                output_path=str(csv_filename),
                include_unlabeled=include_unlabeled
            )

        print(f"✓ Exported {len(videos_data)} videos to {output_dir}")

    def get_statistics(self, labeled_features_list: List[LabeledMotionFeatures]) -> dict:
        """
        Get statistics about the labeled features dataset.

        Args:
            labeled_features_list: List of LabeledMotionFeatures

        Returns:
            Dictionary with statistics about the dataset
        """
        if not labeled_features_list:
            return {
                'total_frames': 0,
                'labeled_frames': 0,
                'unlabeled_frames': 0,
                'videos': [],
                'activity_counts': {}
            }

        # Count labels
        activity_counts = {}
        videos = set()
        labeled_count = 0

        for labeled_feature in labeled_features_list:
            videos.add(labeled_feature.video_id)

            if labeled_feature.has_label():
                labeled_count += 1
                label_name = labeled_feature.activity_label.value
                activity_counts[label_name] = activity_counts.get(label_name, 0) + 1

        return {
            'total_frames': len(labeled_features_list),
            'labeled_frames': labeled_count,
            'unlabeled_frames': len(labeled_features_list) - labeled_count,
            'videos': sorted(list(videos)),
            'activity_counts': activity_counts
        }

    def print_statistics(self, labeled_features_list: List[LabeledMotionFeatures]) -> None:
        """
        Print statistics about the labeled features dataset.

        Args:
            labeled_features_list: List of LabeledMotionFeatures
        """
        stats = self.get_statistics(labeled_features_list)

        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total frames: {stats['total_frames']}")
        print(f"Labeled frames: {stats['labeled_frames']}")
        print(f"Unlabeled frames: {stats['unlabeled_frames']}")
        print(f"Videos: {len(stats['videos'])}")
        print()

        if stats['activity_counts']:
            print("Activity distribution:")
            for activity, count in sorted(stats['activity_counts'].items()):
                percentage = (count / stats['total_frames']) * 100
                print(f"  {activity:30s}: {count:5d} ({percentage:5.2f}%)")
        print("=" * 60 + "\n")
