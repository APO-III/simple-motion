import json
from pathlib import Path
from typing import List, Dict
from labels_reader.domain.video_labels import (
    LabelsDataset,
    VideoAnnotation,
    ActivityLabel
)


class LabelStudioReader:
    """
    Service for reading and parsing Label Studio annotations.
    """

    def __init__(self):
        """Initialize LabelStudioReader."""
        pass

    def _extract_video_id_from_path(self, video_path: str) -> str:
        """
        Extract video ID from Label Studio path format.

        Args:
            video_path: Path in format "/data/upload/3/eea045d1-test.mp4"

        Returns:
            Video ID (e.g., "test")
        """
        filename = video_path.split('/')[-1]

        if '-' in filename:
            video_id = filename.split('-', 1)[1]
            video_id = video_id.rsplit('.', 1)[0]
        else:
            video_id = filename.rsplit('.', 1)[0]

        return video_id

    def _parse_single_task(self, task: dict) -> VideoAnnotation:
        """
        Parse a single Label Studio task into VideoAnnotation.

        Args:
            task: Label Studio task dictionary

        Returns:
            VideoAnnotation object
        """
        video_path = task.get('data', {}).get('video', '')
        video_id = self._extract_video_id_from_path(video_path)

        video_annotation = VideoAnnotation(video_id=video_id)

        annotations = task.get('annotations', [])
        if not annotations:
            return video_annotation

        for annotation in annotations:
            results = annotation.get('result', [])

            for result in results:
                if result.get('type') == 'timelinelabels':
                    value = result.get('value', {})
                    ranges = value.get('ranges', [])
                    timeline_labels = value.get('timelinelabels', [])

                    if timeline_labels and ranges:
                        activity_str = timeline_labels[0]

                        try:
                            activity = ActivityLabel.from_string(activity_str)

                            for range_data in ranges:
                                start_frame = range_data.get('start', 0)
                                end_frame = range_data.get('end', 0)

                                video_annotation.add_frame_range(
                                    start=start_frame,
                                    end=end_frame,
                                    activity=activity
                                )
                        except ValueError as e:
                            print(f"Warning: Skipping unknown activity label '{activity_str}': {e}")

        return video_annotation

    def read_labels_from_json(self, json_path: str) -> LabelsDataset:
        """
        Read Label Studio annotations from JSON file.

        Args:
            json_path: Path to Label Studio export JSON file

        Returns:
            LabelsDataset containing all video annotations
        """
        json_file = Path(json_path)

        if not json_file.exists():
            raise FileNotFoundError(f"Labels file not found: {json_path}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = LabelsDataset()

        if isinstance(data, list):
            for task in data:
                video_annotation = self._parse_single_task(task)
                dataset.add_video_annotation(video_annotation)
        elif isinstance(data, dict):
            video_annotation = self._parse_single_task(data)
            dataset.add_video_annotation(video_annotation)

        return dataset

    def get_video_files_from_directory(self, videos_dir: str) -> Dict[str, str]:
        """
        Get mapping of video IDs to full video paths.

        Args:
            videos_dir: Directory containing video files

        Returns:
            Dictionary mapping video_id to full file path
        """
        videos_path = Path(videos_dir)

        if not videos_path.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = {}

        for video_file in videos_path.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                if '-' in video_file.stem:
                    video_id = video_file.stem.split('-', 1)[1]
                else:
                    video_id = video_file.stem

                video_files[video_id] = str(video_file.absolute())

        return video_files

    def load_dataset(
        self,
        videos_dir: str,
        labels_json_path: str
    ) -> tuple[LabelsDataset, Dict[str, str]]:
        """
        Load complete dataset with labels and video file paths.

        Args:
            videos_dir: Directory containing video files
            labels_json_path: Path to Label Studio JSON file

        Returns:
            Tuple of (LabelsDataset, video_files_dict)
        """
        labels_dataset = self.read_labels_from_json(labels_json_path)
        video_files = self.get_video_files_from_directory(videos_dir)

        return labels_dataset, video_files
