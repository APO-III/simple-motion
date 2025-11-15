"""
Random Forest Training with Hyperparameter Optimization.

This script improves upon the base Random Forest model by:
1. Using GridSearchCV to find optimal hyperparameters
2. Handling class imbalance with balanced weights
3. Cross-validation for robust parameter selection
4. Detailed comparison with baseline results

Usage:
    python example_train_random_forest_optimized.py
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import time

from ml_training.models.random_forest_model import RandomForestMotionClassifier


def load_data_from_csvs(csv_paths, feature_columns, verbose=True):
    """Load and combine data from multiple CSV files."""
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

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"COMBINED DATASET")
        print(f"{'='*80}")
        print(f"Total frames: {len(combined_df)}")
        print(f"Features: {len(feature_columns)}")
        print()

        print("Class distribution:")
        class_counts = combined_df['activity_label'].value_counts()
        for label, count in class_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  {label:30s}: {count:5d} ({percentage:5.1f}%)")
        print()

    X = combined_df[feature_columns].values
    y_labels = combined_df['activity_label'].values

    unique_labels = sorted(combined_df['activity_label'].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    y = np.array([label_to_index[label] for label in y_labels])

    return X, y, label_to_index, index_to_label, combined_df


def split_data(X, y, test_size=0.2, val_size=0.15, random_state=42, verbose=True):
    """Split data into train, validation, and test sets."""
    if verbose:
        print(f"{'='*80}")
        print("SPLITTING DATA")
        print(f"{'='*80}")
        print(f"Test size: {test_size*100:.0f}%")
        print(f"Validation size: {val_size*100:.0f}% (of remaining data)")
        print()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )

    if verbose:
        print(f"Split results:")
        print(f"  Training set:   {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"{'='*80}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


def perform_grid_search(X_train, y_train, verbose=True):
    """
    Perform Grid Search to find optimal hyperparameters.

    Tests different combinations of:
    - n_estimators: number of trees
    - max_depth: maximum depth of trees
    - min_samples_split: minimum samples to split a node
    - min_samples_leaf: minimum samples at leaf node
    - max_features: features to consider for splits
    """
    if verbose:
        print(f"\n{'='*80}")
        print("HYPERPARAMETER OPTIMIZATION WITH GRID SEARCH")
        print(f"{'='*80}\n")

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']  # Always use balanced weights
    }

    if verbose:
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param:20s}: {values}")

        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal combinations to test: {total_combinations}")
        print(f"Using 3-fold cross-validation")
        print(f"Total fits: {total_combinations * 3}")
        print(f"\nThis may take several minutes...")
        print()

    # Create base estimator
    rf_base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='f1_weighted',  # Optimize for weighted F1-score
        n_jobs=-1,
        verbose=2 if verbose else 0,
        return_train_score=True
    )

    print("Starting grid search...")
    start_time = time.time()

    grid_search.fit(X_train, y_train)

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"\n{'='*80}")
        print("GRID SEARCH COMPLETED")
        print(f"{'='*80}")
        print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print()
        print("Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param:20s}: {value}")
        print()
        print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
        print()

        # Show top 5 parameter combinations
        print("Top 5 parameter combinations:")
        print(f"{'Rank':>5s} {'Mean CV F1':>12s} {'Std':>8s} {'Parameters':>40s}")
        print("-" * 80)

        results = pd.DataFrame(grid_search.cv_results_)
        results = results.sort_values('rank_test_score')

        for idx, row in results.head(5).iterrows():
            params_str = str({k: v for k, v in row['params'].items() if k != 'class_weight'})
            if len(params_str) > 40:
                params_str = params_str[:37] + "..."
            print(f"{int(row['rank_test_score']):5d} "
                  f"{row['mean_test_score']:12.4f} "
                  f"{row['std_test_score']:8.4f} "
                  f"{params_str}")
        print()

    return grid_search, grid_search.best_params_, grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, index_to_label, verbose=True):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    report = classification_report(
        y_test, y_pred,
        target_names=[index_to_label[i] for i in sorted(index_to_label.keys())],
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Confidence statistics
    max_probas = np.max(y_proba, axis=1)
    confidence_stats = {
        'mean': float(np.mean(max_probas)),
        'median': float(np.median(max_probas)),
        'std': float(np.std(max_probas)),
        'min': float(np.min(max_probas)),
        'max': float(np.max(max_probas))
    }

    if verbose:
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}\n")

        print(f"Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f} (weighted)")
        print(f"  Recall:    {recall:.4f} (weighted)")
        print(f"  F1-Score:  {f1:.4f} (weighted)")
        print()

        print("Per-Class Metrics:")
        print(f"{'Class':30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("-" * 70)
        for label in sorted(index_to_label.values()):
            metrics = report[label]
            print(f"{label:30s} "
                  f"{metrics['precision']:10.4f} "
                  f"{metrics['recall']:10.4f} "
                  f"{metrics['f1-score']:10.4f} "
                  f"{int(metrics['support']):10d}")
        print()

        print("Prediction Confidence:")
        print(f"  Mean:   {confidence_stats['mean']:.4f}")
        print(f"  Median: {confidence_stats['median']:.4f}")
        print(f"  Std:    {confidence_stats['std']:.4f}")
        print(f"  Range:  [{confidence_stats['min']:.4f}, {confidence_stats['max']:.4f}]")

        print(f"\n{'='*70}\n")

    metrics = {
        'overall_accuracy': accuracy,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1_score': f1,
        'per_class_metrics': report,
        'confusion_matrix': cm.tolist(),
        'confidence_stats': confidence_stats
    }

    return metrics


def main():
    print("\n" + "=" * 80)
    print(" " * 15 + "OPTIMIZED RANDOM FOREST TRAINING PIPELINE")
    print("=" * 80 + "\n")

    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    print("Step 1: Configuration")
    print("-" * 80)

    csv_paths = [
        "results/raw/source1.csv",
        "results/raw/source2.csv",
        "results/raw/source3.csv",
    ]

    feature_columns = [
        'normalized_leg_length',
        'shoulder_vector_x',
        'shoulder_vector_z',
        'ankle_vector_x',
        'ankle_vector_z',
        'average_hip_angle',
        'average_knee_angle'
    ]

    print(f"CSV sources: {len(csv_paths)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Optimization: GridSearchCV with 3-fold cross-validation")
    print(f"Class balancing: Enabled (balanced weights)")
    print()

    # =========================================================================
    # STEP 2: Load Data
    # =========================================================================
    X, y, label_to_index, index_to_label, combined_df = load_data_from_csvs(
        csv_paths=csv_paths,
        feature_columns=feature_columns,
        verbose=True
    )

    # =========================================================================
    # STEP 3: Split Data
    # =========================================================================
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X=X, y=y, test_size=0.20, val_size=0.15, random_state=42, verbose=True
    )

    # =========================================================================
    # STEP 4: Hyperparameter Optimization
    # =========================================================================
    grid_search, best_params, best_model = perform_grid_search(
        X_train=X_train,
        y_train=y_train,
        verbose=True
    )

    # =========================================================================
    # STEP 5: Evaluate on Validation Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Evaluating optimized model on validation set")
    print("=" * 80)

    val_metrics = evaluate_model(
        model=best_model,
        X_test=X_val,
        y_test=y_val,
        index_to_label=index_to_label,
        verbose=True
    )

    # =========================================================================
    # STEP 6: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Evaluating optimized model on test set")
    print("=" * 80)

    test_metrics = evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        index_to_label=index_to_label,
        verbose=True
    )

    # =========================================================================
    # STEP 7: Feature Importance
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Feature Importance Analysis")
    print("=" * 80 + "\n")

    feature_importance = dict(zip(feature_columns, best_model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

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
    # STEP 8: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 8: Saving optimized model and results")
    print("=" * 80 + "\n")

    output_dir = Path("output/random_forest_optimized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters
    with open(output_dir / "best_hyperparameters.json", 'w') as f:
        # Convert any numpy types to Python types
        best_params_serializable = {}
        for k, v in best_params.items():
            if v is None:
                best_params_serializable[k] = None
            elif isinstance(v, (np.integer, np.floating)):
                best_params_serializable[k] = v.item()
            else:
                best_params_serializable[k] = v
        json.dump(best_params_serializable, f, indent=2)
    print("✓ Best hyperparameters saved")

    # Save grid search results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(output_dir / "grid_search_results.csv", index=False)
    print("✓ Grid search results saved")

    # Save metrics
    with open(output_dir / "validation_metrics.json", 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print("✓ Validation metrics saved")

    with open(output_dir / "test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print("✓ Test metrics saved")

    # Save feature importance
    with open(output_dir / "feature_importance.json", 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print("✓ Feature importance saved")

    # Save the optimized model using our wrapper
    optimized_wrapper = RandomForestMotionClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        class_weight=best_params['class_weight'],
        random_state=42,
        n_jobs=-1
    )

    # Copy the trained model
    optimized_wrapper.model = best_model
    optimized_wrapper.label_to_index = label_to_index
    optimized_wrapper.index_to_label = index_to_label
    optimized_wrapper.feature_names = feature_columns

    optimized_wrapper.save(str(output_dir / "random_forest_optimized.pkl"))
    print("✓ Optimized model saved")

    # =========================================================================
    # STEP 9: Compare with Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 9: Comparison with baseline model")
    print("=" * 80 + "\n")

    baseline_path = Path("output/random_forest/test_metrics.json")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)

        print(f"{'Metric':30s} {'Baseline':>12s} {'Optimized':>12s} {'Improvement':>12s}")
        print("-" * 80)

        baseline_acc = baseline_metrics['overall_accuracy']
        optimized_acc = test_metrics['overall_accuracy']
        print(f"{'Test Accuracy':30s} "
              f"{baseline_acc:12.4f} "
              f"{optimized_acc:12.4f} "
              f"{(optimized_acc - baseline_acc):+12.4f}")

        baseline_f1 = baseline_metrics['weighted_f1_score']
        optimized_f1 = test_metrics['weighted_f1_score']
        print(f"{'Test F1-Score':30s} "
              f"{baseline_f1:12.4f} "
              f"{optimized_f1:12.4f} "
              f"{(optimized_f1 - baseline_f1):+12.4f}")

        baseline_prec = baseline_metrics['weighted_precision']
        optimized_prec = test_metrics['weighted_precision']
        print(f"{'Test Precision':30s} "
              f"{baseline_prec:12.4f} "
              f"{optimized_prec:12.4f} "
              f"{(optimized_prec - baseline_prec):+12.4f}")

        baseline_rec = baseline_metrics['weighted_recall']
        optimized_rec = test_metrics['weighted_recall']
        print(f"{'Test Recall':30s} "
              f"{baseline_rec:12.4f} "
              f"{optimized_rec:12.4f} "
              f"{(optimized_rec - baseline_rec):+12.4f}")

        print()
    else:
        print("⚠ Baseline results not found. Run example_train_random_forest.py first.")
        print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("OPTIMIZATION PIPELINE COMPLETED")
    print("=" * 80)

    print(f"\n✓ Hyperparameter search: {len(grid_search.cv_results_['params'])} combinations tested")
    print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")
    print(f"✓ Test accuracy: {test_metrics['overall_accuracy']:.4f}")
    print(f"✓ Test F1-score: {test_metrics['weighted_f1_score']:.4f}")

    print(f"\nGenerated files in {output_dir}:")
    print(f"  - random_forest_optimized.pkl")
    print(f"  - best_hyperparameters.json")
    print(f"  - grid_search_results.csv")
    print(f"  - validation_metrics.json")
    print(f"  - test_metrics.json")
    print(f"  - feature_importance.json")

    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
