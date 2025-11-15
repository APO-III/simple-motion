"""
Random Forest model for motion classification.

This module provides a Random Forest classifier that works with
individual frames (no temporal sequences required).
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class RandomForestMotionClassifier:
    """
    Random Forest classifier for motion activity recognition.

    Unlike LSTM, this model works with individual frames rather than
    temporal sequences. It's simpler, faster to train, and often performs
    well with small datasets.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1,
        class_weight: str = 'balanced'
    ):
        """
        Initialize Random Forest classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = use all cores)
            class_weight: 'balanced' to handle imbalanced classes
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
            verbose=0
        )

        self.label_to_index: Optional[Dict[str, int]] = None
        self.index_to_label: Optional[Dict[int, str]] = None
        self.feature_names: Optional[list] = None
        self.is_trained = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_to_index: Dict[str, int],
        index_to_label: Dict[int, str],
        feature_names: Optional[list] = None,
        verbose: bool = True
    ):
        """
        Train the Random Forest model.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels (integer encoded), shape (n_samples,)
            label_to_index: Mapping from label names to indices
            index_to_label: Mapping from indices to label names
            feature_names: Optional list of feature names
            verbose: Print training progress
        """
        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING RANDOM FOREST")
            print(f"{'='*70}")
            print(f"Training samples: {len(X)}")
            print(f"Number of features: {X.shape[1]}")
            print(f"Number of classes: {len(label_to_index)}")
            print(f"Number of trees: {self.model.n_estimators}")
            print(f"Class weight: {self.model.class_weight}")
            print()

        # Store label mappings
        self.label_to_index = label_to_index
        self.index_to_label = index_to_label
        self.feature_names = feature_names

        # Train model
        self.model.fit(X, y)
        self.is_trained = True

        if verbose:
            print("✓ Training completed")
            print(f"{'='*70}\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Predicted class indices, shape (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Predicted probabilities, shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features, shape (n_samples, n_features)
            y_test: Test labels (integer encoded), shape (n_samples,)
            verbose: Print evaluation results

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # Get all class labels
        all_labels = list(range(len(self.label_to_index)))

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=all_labels, average=None, zero_division=0
        )

        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=all_labels, average='weighted', zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred, labels=all_labels)

        # Per-class metrics
        per_class_metrics = {}
        for idx in range(len(self.label_to_index)):
            label_name = self.index_to_label[idx]
            per_class_metrics[label_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx])
            }

        # Prediction confidence
        max_probs = np.max(y_pred_proba, axis=1)
        confidence_stats = {
            "mean_confidence": float(np.mean(max_probs)),
            "median_confidence": float(np.median(max_probs)),
            "min_confidence": float(np.min(max_probs)),
            "max_confidence": float(np.max(max_probs)),
            "std_confidence": float(np.std(max_probs))
        }

        metrics = {
            "overall_accuracy": float(accuracy),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1_score": float(f1_weighted),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "confidence_stats": confidence_stats,
            "num_samples": int(len(y_test))
        }

        if verbose:
            self._print_evaluation(metrics)

        return metrics

    def _print_evaluation(self, metrics: Dict):
        """Print evaluation metrics in readable format."""
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")

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
            print(f"{label_name:30s} {m['precision']:10.4f} {m['recall']:10.4f} "
                  f"{m['f1_score']:10.4f} {m['support']:10d}")

        print(f"\nPrediction Confidence:")
        conf = metrics['confidence_stats']
        print(f"  Mean:   {conf['mean_confidence']:.4f}")
        print(f"  Median: {conf['median_confidence']:.4f}")
        print(f"  Std:    {conf['std_confidence']:.4f}")
        print(f"  Range:  [{conf['min_confidence']:.4f}, {conf['max_confidence']:.4f}]")

        print(f"\nConfusion Matrix:")
        self._print_confusion_matrix(metrics['confusion_matrix'])
        print(f"{'='*70}\n")

    def _print_confusion_matrix(self, cm: list):
        """Print confusion matrix."""
        cm_array = np.array(cm)
        labels = [self.index_to_label[i] for i in range(len(self.label_to_index))]

        # Print header
        print(f"\n{'True \\ Pred':20s}", end='')
        for label in labels:
            print(f"{label[:8]:>8s}", end='')
        print()
        print("-" * (20 + 8 * len(labels)))

        # Print rows
        for i, label in enumerate(labels):
            print(f"{label:20s}", end='')
            for j in range(len(labels)):
                print(f"{cm_array[i, j]:8d}", end='')
            print()

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        importances = self.model.feature_importances_

        if self.feature_names:
            return {
                name: float(importance)
                for name, importance in zip(self.feature_names, importances)
            }
        else:
            return {
                f"feature_{i}": float(importance)
                for i, importance in enumerate(importances)
            }

    def save(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'label_to_index': self.label_to_index,
            'index_to_label': self.index_to_label,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'RandomForestMotionClassifier':
        """
        Load model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded RandomForestMotionClassifier instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance with dummy parameters (will be overwritten)
        instance = cls()
        instance.model = model_data['model']
        instance.label_to_index = model_data['label_to_index']
        instance.index_to_label = model_data['index_to_label']
        instance.feature_names = model_data['feature_names']
        instance.is_trained = model_data['is_trained']

        print(f"✓ Model loaded from {filepath}")
        return instance
