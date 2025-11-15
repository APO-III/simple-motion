"""
Example script to train LSTM model for motion activity classification.

This demonstrates the complete training pipeline:
1. Generate sequences from CSV
2. Split into train/val/test
3. Configure and build LSTM model
4. Train with callbacks
5. Evaluate performance
6. Save model and results

Usage:
    python example_train_lstm.py
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ml_training.domain.sequence import SequenceGeneratorConfig
from ml_training.domain.training_config import (
    LSTMArchitectureConfig,
    TrainingHyperparameters,
    CallbacksConfig,
    TrainingConfig
)
from ml_training.use_cases.sequence_generator import SequenceGenerator
from ml_training.utils.data_splitter import DataSplitter
from ml_training.use_cases.lstm_trainer import LSTMTrainer
from ml_training.use_cases.model_evaluator import ModelEvaluator


def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "LSTM TRAINING PIPELINE")
    print("=" * 80 + "\n")

    # =========================================================================
    # STEP 1: Generate Sequences
    # =========================================================================
    print("Step 1: Generating sequences from CSV files")
    print("-" * 80)

    sequence_config = SequenceGeneratorConfig(
        window_size=30,
        stride=15,
        min_segment_length=30
    )

    generator = SequenceGenerator(config=sequence_config)

    # Generate from multiple sources
    dataset = generator.generate_from_multiple_csvs(
        csv_paths=[
            ("results/raw/source1.csv", "source1"),
            ("results/raw/source2.csv", "source2"),  # RESTORED - critical for dataset size
            ("results/raw/source3.csv", "source3"),
        ],
        verbose=True
    )

    dataset.print_statistics()

    # =========================================================================
    # STEP 2: Split Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Splitting into train/test sets")
    print("=" * 80)

    splitter = DataSplitter(random_seed=42)

    # Since we have few samples, use 80/20 split (no separate validation)
    train_dataset, _, test_dataset = splitter.split_by_video(
        dataset=dataset,
        train_ratio=0.80,
        val_ratio=0.00,  # No validation set, use validation_split instead
        test_ratio=0.20,
        stratify_by_label=True,
        verbose=True
    )

    # =========================================================================
    # STEP 3: Calculate Class Weights (for imbalanced data)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Calculating class weights")
    print("=" * 80 + "\n")

    y_train = train_dataset.get_y()
    classes = np.unique(y_train)

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, class_weights_array)}

    print("Class weights (for handling imbalanced data):")
    for cls_idx, weight in class_weights.items():
        label_name = dataset.index_to_label[cls_idx]
        print(f"  {label_name:30s}: {weight:.3f}")

    # =========================================================================
    # STEP 4: Configure Training
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Configuring LSTM model and training")
    print("=" * 80)

    # Architecture configuration (reduced complexity for small dataset)
    arch_config = LSTMArchitectureConfig(
        lstm1_units=64,      # Reduced from 128 to prevent overfitting
        lstm2_units=32,      # Reduced from 64
        dense_units=16,      # Reduced from 32
        dropout_lstm=0.4,    # Increased from 0.3 for better regularization
        dropout_dense=0.5,   # Increased from 0.2
        recurrent_dropout=0.3,  # Increased from 0.2
        input_shape=(30, 7),
        num_classes=len(dataset.label_to_index)
    )

    # Training hyperparameters (optimized for small dataset)
    hyperparams = TrainingHyperparameters(
        epochs=100,  # Increased to allow model to converge better
        batch_size=8,  # Reduced from 16 for better gradient estimates
        learning_rate=0.0005,  # Reduced from 0.001 for more stable training
        validation_split=0.15,  # Use 15% of training data for validation
        class_weights=class_weights,
        shuffle=True,
        verbose=1
    )

    # Callbacks configuration (optimized for imbalanced dataset)
    callbacks_config = CallbacksConfig(
        use_early_stopping=True,
        early_stopping_patience=15,  # Increased from 10 to allow more training
        early_stopping_monitor="val_accuracy",  # Changed from val_loss to focus on accuracy
        restore_best_weights=True,

        use_reduce_lr=True,
        reduce_lr_patience=7,  # Increased from 5
        reduce_lr_factor=0.5,

        use_model_checkpoint=True,
        checkpoint_filepath="output/models/lstm_motion_classifier.keras",
        checkpoint_save_best_only=True,

        use_tensorboard=True,
        tensorboard_log_dir="output/logs/tensorboard"
    )

    # Complete training configuration
    training_config = TrainingConfig(
        architecture=arch_config,
        hyperparameters=hyperparams,
        callbacks=callbacks_config
    )

    training_config.print_summary()

    # =========================================================================
    # STEP 5: Build Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Building LSTM model")
    print("=" * 80)

    trainer = LSTMTrainer(config=training_config)
    model_wrapper = trainer.build_model()

    # =========================================================================
    # STEP 6: Train Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Training LSTM model")
    print("=" * 80)

    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=None,  # Using validation_split instead
        verbose=True
    )

    # Save training history
    trainer.save_history("output/training_history.json")

    # =========================================================================
    # STEP 7: Evaluate Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Evaluating model on test set")
    print("=" * 80)

    evaluator = ModelEvaluator(
        model=model_wrapper,
        label_to_index=dataset.label_to_index,
        index_to_label=dataset.index_to_label
    )

    metrics = evaluator.evaluate(
        test_dataset=test_dataset,
        verbose=True
    )

    # Save evaluation report
    evaluator.save_evaluation_report(metrics, "output/evaluation_report.json")

    # =========================================================================
    # STEP 8: Save Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 8: Saving trained model")
    print("=" * 80 + "\n")

    trainer.save_model("output/models/lstm_motion_classifier_final.keras")

    # Also save label encoder (if not already saved)
    from ml_training.utils.label_encoder import LabelEncoder
    encoder = LabelEncoder(
        label_to_index=dataset.label_to_index,
        index_to_label=dataset.index_to_label
    )
    encoder.save("output/label_encoder.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 80)

    print(f"\n✓ Model trained on {len(train_dataset)} sequences")
    print(f"✓ Model evaluated on {len(test_dataset)} sequences")
    print(f"✓ Test accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"✓ Test F1-score: {metrics['weighted_f1_score']:.4f}")

    print(f"\nGenerated files:")
    print(f"  - output/models/lstm_motion_classifier_final.keras (trained model)")
    print(f"  - output/models/lstm_motion_classifier_final_config.json (model config)")
    print(f"  - output/label_encoder.json (label encoder)")
    print(f"  - output/training_history.json (training metrics)")
    print(f"  - output/evaluation_report.json (test metrics)")
    print(f"  - output/logs/tensorboard/* (TensorBoard logs)")

    print(f"\nNext steps:")
    print(f"  1. Visualize training: tensorboard --logdir=output/logs/tensorboard")
    print(f"  2. Load model for inference: KerasLSTMModel.load('output/models/...')")
    print(f"  3. Process more videos to increase dataset size")
    print(f"  4. Experiment with hyperparameters for better performance")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
