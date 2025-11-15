"""
Model evaluation use case.

This module provides comprehensive evaluation metrics for trained LSTM models:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- Per-class metrics
- Prediction analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import json
from pathlib import Path

from ml_training.domain.sequence import SequenceDataset
from ml_training.infrastructure.keras_lstm_model import KerasLSTMModel


class ModelEvaluator:
    """
    Evaluates LSTM model performance on test data.

    Provides comprehensive metrics including:
    - Overall accuracy
    - Per-class precision, recall, F1-score
    - Confusion matrix
    - Misclassification analysis
    """

    def __init__(self, model: KerasLSTMModel, label_to_index: Dict[str, int], index_to_label: Dict[int, str]):
        """
        Initialize model evaluator.

        Args:
            model: Trained LSTM model
            label_to_index: Mapping from label names to indices
            index_to_label: Mapping from indices to label names
        """
        self.model = model
        self.label_to_index = label_to_index
        self.index_to_label = index_to_label
        self.num_classes = len(label_to_index)

    def evaluate(
        self,
        test_dataset: SequenceDataset,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model on test dataset.

        Args:
            test_dataset: Test dataset
            verbose: Print evaluation results

        Returns:
            Dictionary with evaluation metrics
        """
        # Get test data
        X_test = test_dataset.get_X()
        y_test = test_dataset.get_y()

        if verbose:
            print("\n" + "=" * 70)
            print("MODEL EVALUATION")
            print("=" * 70 + "\n")
            print(f"Test data:")
            print(f"  Samples: {len(X_test)}")
            print(f"  Shape: {X_test.shape}")
            print()

        # Make predictions
        y_pred_proba = self.model.get_model().predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        if verbose:
            self._print_metrics(metrics)

        return metrics

    def evaluate_with_arrays(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model with NumPy arrays.

        Args:
            X_test: Test features (num_samples, timesteps, features)
            y_test: Test labels (num_samples,) - integer encoded
            verbose: Print evaluation results

        Returns:
            Dictionary with evaluation metrics
        """
        if verbose:
            print("\n" + "=" * 70)
            print("MODEL EVALUATION")
            print("=" * 70 + "\n")
            print(f"Test data:")
            print(f"  Samples: {len(X_test)}")
            print(f"  Shape: {X_test.shape}")
            print()

        # Make predictions
        y_pred_proba = self.model.get_model().predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        if verbose:
            self._print_metrics(metrics)

        return metrics

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels (integer encoded)
            y_pred: Predicted labels (integer encoded)
            y_pred_proba: Prediction probabilities

        Returns:
            Dictionary with all metrics
        """
        # Overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)

        # Get all class labels
        all_labels = list(range(self.num_classes))

        # Per-class metrics (ensure all classes are included)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )

        # Weighted average metrics
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average='weighted', zero_division=0
        )

        # Confusion matrix (ensure all classes are included)
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        # Per-class metrics dict
        per_class_metrics = {}
        for idx in range(self.num_classes):
            label_name = self.index_to_label[idx]
            per_class_metrics[label_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx])
            }

        # Prediction confidence
        confidence_stats = self._calculate_confidence_stats(y_pred_proba)

        return {
            "overall_accuracy": float(overall_accuracy),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1_score": float(f1_weighted),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "confidence_stats": confidence_stats,
            "num_samples": int(len(y_true))
        }

    def _calculate_confidence_stats(self, y_pred_proba: np.ndarray) -> Dict:
        """Calculate statistics on prediction confidence."""
        max_probs = np.max(y_pred_proba, axis=1)

        return {
            "mean_confidence": float(np.mean(max_probs)),
            "median_confidence": float(np.median(max_probs)),
            "min_confidence": float(np.min(max_probs)),
            "max_confidence": float(np.max(max_probs)),
            "std_confidence": float(np.std(max_probs))
        }

    def _print_metrics(self, metrics: Dict):
        """Print evaluation metrics in a readable format."""
        print("=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['overall_accuracy']:.4f}")
        print(f"  Precision: {metrics['weighted_precision']:.4f} (weighted)")
        print(f"  Recall:    {metrics['weighted_recall']:.4f} (weighted)")
        print(f"  F1-Score:  {metrics['weighted_f1_score']:.4f} (weighted)")

        print(f"\nPer-Class Metrics:")
        print(f"{'Class':30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("-" * 70)

        for label_name in sorted(metrics['per_class_metrics'].keys()):
            m = metrics['per_class_metrics'][label_name]
            print(f"{label_name:30s} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1_score']:10.4f} {m['support']:10d}")

        print(f"\nPrediction Confidence:")
        conf = metrics['confidence_stats']
        print(f"  Mean:   {conf['mean_confidence']:.4f}")
        print(f"  Median: {conf['median_confidence']:.4f}")
        print(f"  Std:    {conf['std_confidence']:.4f}")
        print(f"  Range:  [{conf['min_confidence']:.4f}, {conf['max_confidence']:.4f}]")

        print(f"\nConfusion Matrix:")
        self._print_confusion_matrix(metrics['confusion_matrix'])

        print("=" * 70 + "\n")

    def _print_confusion_matrix(self, cm: List[List[int]]):
        """Print confusion matrix in a readable format."""
        cm_array = np.array(cm)

        # Get label names
        labels = [self.index_to_label[i] for i in range(self.num_classes)]

        # Print header
        print(f"\n{'True \\ Pred':20s}", end='')
        for label in labels:
            print(f"{label[:8]:>8s}", end='')
        print()
        print("-" * (20 + 8 * self.num_classes))

        # Print rows
        for i, label in enumerate(labels):
            print(f"{label:20s}", end='')
            for j in range(self.num_classes):
                print(f"{cm_array[i, j]:8d}", end='')
            print()

    def find_misclassifications(
        self,
        test_dataset: SequenceDataset,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Find and analyze the most confident misclassifications.

        Args:
            test_dataset: Test dataset
            top_n: Number of top misclassifications to return

        Returns:
            List of misclassification details
        """
        X_test = test_dataset.get_X()
        y_test = test_dataset.get_y()

        # Make predictions
        y_pred_proba = self.model.get_model().predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Find misclassifications
        misclassified_indices = np.where(y_true != y_pred)[0]

        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return []

        # Get confidence for misclassified samples
        misclassified_confidence = np.max(y_pred_proba[misclassified_indices], axis=1)

        # Sort by confidence (descending) - most confident mistakes
        sorted_indices = misclassified_indices[np.argsort(-misclassified_confidence)]

        # Get top N
        top_indices = sorted_indices[:top_n]

        misclassifications = []
        for idx in top_indices:
            seq = test_dataset.sequences[idx]
            true_label = self.index_to_label[y_test[idx]]
            pred_label = self.index_to_label[y_pred[idx]]
            confidence = float(np.max(y_pred_proba[idx]))

            misclassifications.append({
                "index": int(idx),
                "video_id": seq.video_id,
                "frames": f"{seq.start_frame}-{seq.end_frame}",
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidence,
                "probabilities": {
                    self.index_to_label[i]: float(y_pred_proba[idx, i])
                    for i in range(self.num_classes)
                }
            })

        return misclassifications

    def save_evaluation_report(self, metrics: Dict, filepath: str = "output/evaluation_report.json"):
        """
        Save evaluation metrics to JSON file.

        Args:
            metrics: Evaluation metrics dictionary
            filepath: Path to save report
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Evaluation report saved to {filepath}")
