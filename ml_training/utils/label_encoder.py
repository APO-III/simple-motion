"""
Label encoding utilities.

Converts activity labels (strings) to integer indices and vice versa.
"""

from typing import Dict, List
import json
from pathlib import Path


class LabelEncoder:
    """
    Encodes activity labels as integers.

    This is a simple wrapper around dictionary mappings that can be
    persisted to disk for model deployment.
    """

    def __init__(self, label_to_index: Dict[str, int], index_to_label: Dict[int, str]):
        """
        Initialize label encoder.

        Args:
            label_to_index: Mapping from label string to integer index
            index_to_label: Mapping from integer index to label string
        """
        self.label_to_index = label_to_index
        self.index_to_label = index_to_label
        self.num_classes = len(label_to_index)

    @classmethod
    def from_labels(cls, labels: List[str]) -> 'LabelEncoder':
        """
        Create encoder from a list of label strings.

        Args:
            labels: List of label strings

        Returns:
            LabelEncoder instance
        """
        unique_labels = sorted(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}

        return cls(label_to_index=label_to_index, index_to_label=index_to_label)

    def encode(self, label: str) -> int:
        """
        Encode a label string to integer.

        Args:
            label: Label string

        Returns:
            Integer index

        Raises:
            KeyError: If label is unknown
        """
        if label not in self.label_to_index:
            raise KeyError(f"Unknown label: {label}. Known labels: {list(self.label_to_index.keys())}")

        return self.label_to_index[label]

    def decode(self, index: int) -> str:
        """
        Decode an integer index to label string.

        Args:
            index: Integer index

        Returns:
            Label string

        Raises:
            KeyError: If index is unknown
        """
        if index not in self.index_to_label:
            raise KeyError(f"Unknown index: {index}. Valid range: 0-{self.num_classes-1}")

        return self.index_to_label[index]

    def encode_batch(self, labels: List[str]) -> List[int]:
        """Encode multiple labels."""
        return [self.encode(label) for label in labels]

    def decode_batch(self, indices: List[int]) -> List[str]:
        """Decode multiple indices."""
        return [self.decode(idx) for idx in indices]

    def save(self, filepath: str):
        """
        Save encoder to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'label_to_index': self.label_to_index,
            'index_to_label': {str(k): v for k, v in self.index_to_label.items()},
            'num_classes': self.num_classes
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved label encoder to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LabelEncoder':
        """
        Load encoder from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            LabelEncoder instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        label_to_index = data['label_to_index']
        index_to_label = {int(k): v for k, v in data['index_to_label'].items()}

        return cls(label_to_index=label_to_index, index_to_label=index_to_label)

    def __repr__(self) -> str:
        return f"LabelEncoder(num_classes={self.num_classes}, labels={list(self.label_to_index.keys())})"
