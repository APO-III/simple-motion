"""
Example script to train Random Forest model for motion activity classification.

Unlike LSTM which uses temporal sequences, Random Forest works with
individual frames. This approach:
- Is simpler and faster to train
- Works well with small datasets
- Provides interpretable feature importance
- Good baseline to compare against deep learning

This demonstrates the complete training pipeline:
1. Load frame-by-frame data from CSV
2. Split into train/val/test
3. Train Random Forest classifier
4. Evaluate performance
5. Save model and results

Usage:
    python example_train_random_forest.py
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ml_training.models.random_forest_model import RandomForestMotionClassifier


def load_data_from_csvs(csv_paths, feature_columns, verbose=True):
    """
    Load and combine data from multiple CSV files.

    Args:
        csv_paths: List of CSV file paths
        feature_columns: List of feature column names
        verbose: Print progress

    Returns:
        Tuple of (X, y, label_to_index, index_to_label, all_data)
    """
    if verbose:
        print(f"\n{'='*80}")
        print("LOADING DATA FROM CSV FILES")
        print(f"{'='*80}\n")

    all_dfs = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"✓ Loaded {len(df)} frames from {Path(csv_path).name}")
        all_dfs.append(df)

    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"COMBINED DATASET")
        print(f"{'='*80}")
        print(f"Total frames: {len(combined_df)}")
        print(f"Features: {len(feature_columns)}")
        print()

        # Print class distribution
        print("Class distribution:")
        class_counts = combined_df['activity_label'].value_counts()
        for label, count in class_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  {label:30s}: {count:5d} ({percentage:5.1f}%)")
        print()

    # Extract features and labels
    X = combined_df[feature_columns].values
    y_labels = combined_df['activity_label'].values

    # Create label mappings
    unique_labels = sorted(combined_df['activity_label'].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    # Convert labels to integers
    y = np.array([label_to_index[label] for label in y_labels])

    return X, y, label_to_index, index_to_label, combined_df


def split_data(X, y, combined_df, test_size=0.2, val_size=0.15, random_state=42, verbose=True):
    """
    Split data into train, validation, and test sets.

    Uses stratified split to maintain class distribution.

    Args:
        X: Features array
        y: Labels array (integer encoded)
        combined_df: Original DataFrame (for video-based splitting if needed)
        test_size: Proportion for test set
        val_size: Proportion of remaining data for validation
        random_state: Random seed
        verbose: Print statistics

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if verbose:
        print(f"{'='*80}")
        print("SPLITTING DATA")
        print(f"{'='*80}")
        print(f"Split strategy: Stratified random split")
        print(f"Test size: {test_size*100:.0f}%")
        print(f"Validation size: {val_size*100:.0f}% (of remaining data)")
        print()

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: separate train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state
    )

    if verbose:
        print(f"Split results:")
        print(f"  Training set:   {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"{'='*80}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "RANDOM FOREST TRAINING PIPELINE")
    print("=" * 80 + "\n")

    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    print("Step 1: Configuration")
    print("-" * 80)

    # CSV files to load
    csv_paths = [
        "results/raw/source1.csv",
        "results/raw/source2.csv",
        "results/raw/source3.csv",
    ]

    # Feature columns (excluding video_id, frame_number, activity_label)
    feature_columns = [
        'normalized_leg_length',
        'shoulder_vector_x',
        'shoulder_vector_z',
        'ankle_vector_x',
        'ankle_vector_z',
        'average_hip_angle',
        'average_knee_angle'
    ]

    # Random Forest hyperparameters
    rf_params = {
        'n_estimators': 200,      # Number of trees
        'max_depth': 20,           # Maximum tree depth
        'min_samples_split': 5,    # Minimum samples to split node
        'min_samples_leaf': 2,     # Minimum samples at leaf
        'max_features': 'sqrt',    # Features to consider for split
        'random_state': 42,
        'n_jobs': -1,              # Use all CPU cores
        'class_weight': 'balanced' # Handle imbalanced classes
    }

    print(f"CSV sources: {len(csv_paths)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Random Forest config: {rf_params['n_estimators']} trees, "
          f"max_depth={rf_params['max_depth']}, class_weight={rf_params['class_weight']}")
    print()

    # =========================================================================
    # STEP 2: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Loading data from CSV files")
    print("=" * 80)

    X, y, label_to_index, index_to_label, combined_df = load_data_from_csvs(
        csv_paths=csv_paths,
        feature_columns=feature_columns,
        verbose=True
    )

    # =========================================================================
    # STEP 3: Split Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Splitting data into train/val/test sets")
    print("=" * 80 + "\n")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X=X,
        y=y,
        combined_df=combined_df,
        test_size=0.20,
        val_size=0.15,
        random_state=42,
        verbose=True
    )

    # =========================================================================
    # STEP 4: Train Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Training Random Forest model")
    print("=" * 80)

    model = RandomForestMotionClassifier(**rf_params)

    model.fit(
        X=X_train,
        y=y_train,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        feature_names=feature_columns,
        verbose=True
    )

    # =========================================================================
    # STEP 5: Evaluate on Validation Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Evaluating on validation set")
    print("=" * 80)

    val_metrics = model.evaluate(
        X_test=X_val,
        y_test=y_val,
        verbose=True
    )

    # =========================================================================
    # STEP 6: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Evaluating on test set")
    print("=" * 80)

    test_metrics = model.evaluate(
        X_test=X_test,
        y_test=y_test,
        verbose=True
    )

    # =========================================================================
    # STEP 7: Feature Importance
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Feature Importance Analysis")
    print("=" * 80 + "\n")

    feature_importance = model.get_feature_importance()
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Feature Importance (sorted by importance):")
    print(f"{'Feature':35s} {'Importance':>12s} {'Bar':>20s}")
    print("-" * 80)

    max_importance = max(feature_importance.values())
    for feature, importance in sorted_features:
        bar_length = int((importance / max_importance) * 20)
        bar = '█' * bar_length
        print(f"{feature:35s} {importance:12.4f} {bar:>20s}")
    print()

    # =========================================================================
    # STEP 8: Save Model and Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 8: Saving model and results")
    print("=" * 80 + "\n")

    # Create output directory
    output_dir = Path("output/random_forest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save("output/random_forest/random_forest_classifier.pkl")

    # Save validation metrics
    with open("output/random_forest/validation_metrics.json", 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print("✓ Validation metrics saved to output/random_forest/validation_metrics.json")

    # Save test metrics
    with open("output/random_forest/test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print("✓ Test metrics saved to output/random_forest/test_metrics.json")

    # Save feature importance
    with open("output/random_forest/feature_importance.json", 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print("✓ Feature importance saved to output/random_forest/feature_importance.json")

    # Save label mappings
    label_mappings = {
        'label_to_index': label_to_index,
        'index_to_label': index_to_label
    }
    with open("output/random_forest/label_mappings.json", 'w') as f:
        json.dump(label_mappings, f, indent=2)
    print("✓ Label mappings saved to output/random_forest/label_mappings.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 80)

    print(f"\n✓ Model trained on {len(X_train)} frames")
    print(f"✓ Validated on {len(X_val)} frames")
    print(f"✓ Tested on {len(X_test)} frames")
    print(f"✓ Validation accuracy: {val_metrics['overall_accuracy']:.4f}")
    print(f"✓ Test accuracy: {test_metrics['overall_accuracy']:.4f}")
    print(f"✓ Test F1-score: {test_metrics['weighted_f1_score']:.4f}")

    print(f"\nGenerated files:")
    print(f"  - output/random_forest/random_forest_classifier.pkl")
    print(f"  - output/random_forest/validation_metrics.json")
    print(f"  - output/random_forest/test_metrics.json")
    print(f"  - output/random_forest/feature_importance.json")
    print(f"  - output/random_forest/label_mappings.json")

    print(f"\nNext steps:")
    print(f"  1. Compare with LSTM results")
    print(f"  2. Analyze feature importance to understand model")
    print(f"  3. Try feature engineering based on importance")
    print(f"  4. Load model for inference: RandomForestMotionClassifier.load('output/...')")

    print("\n" + "=" * 80 + "\n")

    # Print comparison with LSTM if available
    lstm_results_path = Path("output/evaluation_report.json")
    if lstm_results_path.exists():
        print("\n" + "=" * 80)
        print("COMPARISON: Random Forest vs LSTM")
        print("=" * 80 + "\n")

        with open(lstm_results_path, 'r') as f:
            lstm_metrics = json.load(f)

        print(f"{'Metric':30s} {'Random Forest':>15s} {'LSTM':>15s} {'Difference':>15s}")
        print("-" * 80)

        rf_acc = test_metrics['overall_accuracy']
        lstm_acc = lstm_metrics['overall_accuracy']
        print(f"{'Test Accuracy':30s} {rf_acc:15.4f} {lstm_acc:15.4f} {rf_acc - lstm_acc:+15.4f}")

        rf_f1 = test_metrics['weighted_f1_score']
        lstm_f1 = lstm_metrics['weighted_f1_score']
        print(f"{'Test F1-Score':30s} {rf_f1:15.4f} {lstm_f1:15.4f} {rf_f1 - lstm_f1:+15.4f}")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
