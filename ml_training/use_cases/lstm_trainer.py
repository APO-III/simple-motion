"""
LSTM trainer use case.

This module handles the complete training pipeline for LSTM models:
- Model compilation
- Callbacks setup
- Training execution
- History tracking
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime
import json

from tensorflow import keras

from ml_training.domain.training_config import TrainingConfig
from ml_training.domain.sequence import SequenceDataset
from ml_training.infrastructure.keras_lstm_model import KerasLSTMModel


class LSTMTrainer:
    """
    Trains LSTM models for motion activity classification.

    This service orchestrates the entire training process including:
    - Model initialization and compilation
    - Callbacks configuration
    - Training execution
    - Model persistence
    - Training history tracking
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize LSTM trainer.

        Args:
            config: Complete training configuration
        """
        self.config = config
        self.model_wrapper: Optional[KerasLSTMModel] = None
        self.history: Optional[keras.callbacks.History] = None

    def build_model(self) -> KerasLSTMModel:
        """
        Build and compile LSTM model.

        Returns:
            Compiled KerasLSTMModel instance
        """
        print("\n" + "=" * 70)
        print("BUILDING LSTM MODEL")
        print("=" * 70 + "\n")

        # Build model
        self.model_wrapper = KerasLSTMModel(self.config.architecture)

        # Compile model
        self.model_wrapper.compile_model(
            learning_rate=self.config.hyperparameters.learning_rate
        )

        # Print summary
        self.model_wrapper.print_architecture_summary()

        return self.model_wrapper

    def _setup_callbacks(self) -> list:
        """
        Setup training callbacks based on configuration.

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Early Stopping
        if self.config.callbacks.use_early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor=self.config.callbacks.early_stopping_monitor,
                patience=self.config.callbacks.early_stopping_patience,
                min_delta=self.config.callbacks.early_stopping_min_delta,
                restore_best_weights=self.config.callbacks.restore_best_weights,
                verbose=1
            )
            callbacks.append(early_stop)
            print("✓ Early stopping enabled")

        # Reduce Learning Rate on Plateau
        if self.config.callbacks.use_reduce_lr:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor=self.config.callbacks.early_stopping_monitor,
                patience=self.config.callbacks.reduce_lr_patience,
                factor=self.config.callbacks.reduce_lr_factor,
                min_lr=self.config.callbacks.reduce_lr_min_lr,
                verbose=1
            )
            callbacks.append(reduce_lr)
            print("✓ Reduce LR on plateau enabled")

        # Model Checkpoint
        if self.config.callbacks.use_model_checkpoint:
            checkpoint_path = Path(self.config.callbacks.checkpoint_filepath)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.config.callbacks.early_stopping_monitor,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=1
            )
            callbacks.append(checkpoint)
            print(f"✓ Model checkpoint enabled: {checkpoint_path}")

        # TensorBoard
        if self.config.callbacks.use_tensorboard:
            log_dir = Path(self.config.callbacks.tensorboard_log_dir)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = log_dir / timestamp
            log_dir.mkdir(parents=True, exist_ok=True)

            tensorboard = keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True
            )
            callbacks.append(tensorboard)
            print(f"✓ TensorBoard enabled: {log_dir}")

        return callbacks

    def train(
        self,
        train_dataset: SequenceDataset,
        val_dataset: Optional[SequenceDataset] = None,
        verbose: bool = True
    ) -> keras.callbacks.History:
        """
        Train the LSTM model.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            verbose: Print training progress

        Returns:
            Training history

        Raises:
            ValueError: If model hasn't been built
        """
        if self.model_wrapper is None:
            raise ValueError("Model not built. Call build_model() first.")

        if verbose:
            print("\n" + "=" * 70)
            print("STARTING TRAINING")
            print("=" * 70 + "\n")

        # Prepare data
        X_train = train_dataset.get_X()
        y_train = train_dataset.get_y_categorical(self.config.architecture.num_classes)

        if verbose:
            print(f"Training data:")
            print(f"  X_train shape: {X_train.shape}")
            print(f"  y_train shape: {y_train.shape}")

        # Prepare validation data
        validation_data = None
        if val_dataset is not None and len(val_dataset) > 0:
            X_val = val_dataset.get_X()
            y_val = val_dataset.get_y_categorical(self.config.architecture.num_classes)
            validation_data = (X_val, y_val)

            if verbose:
                print(f"\nValidation data:")
                print(f"  X_val shape: {X_val.shape}")
                print(f"  y_val shape: {y_val.shape}")
        else:
            if verbose:
                print(f"\nNo validation dataset provided.")
                if self.config.hyperparameters.validation_split > 0:
                    print(f"Using validation_split={self.config.hyperparameters.validation_split}")

        # Setup callbacks
        if verbose:
            print("\nSetting up callbacks:")

        callbacks = self._setup_callbacks()

        if verbose:
            print(f"\nTraining configuration:")
            print(f"  Epochs: {self.config.hyperparameters.epochs}")
            print(f"  Batch size: {self.config.hyperparameters.batch_size}")
            print(f"  Learning rate: {self.config.hyperparameters.learning_rate}")
            print(f"  Class weights: {self.config.hyperparameters.class_weights is not None}")
            print("\n" + "=" * 70)
            print()

        # Train model
        self.history = self.model_wrapper.get_model().fit(
            X_train,
            y_train,
            validation_data=validation_data,
            validation_split=self.config.hyperparameters.validation_split if validation_data is None else 0.0,
            epochs=self.config.hyperparameters.epochs,
            batch_size=self.config.hyperparameters.batch_size,
            class_weight=self.config.hyperparameters.class_weights,
            shuffle=self.config.hyperparameters.shuffle,
            callbacks=callbacks,
            verbose=self.config.hyperparameters.verbose
        )

        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED")
            print("=" * 70)
            self._print_training_summary()

        return self.history

    def train_with_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> keras.callbacks.History:
        """
        Train the LSTM model with NumPy arrays.

        Args:
            X_train: Training features (num_samples, timesteps, features)
            y_train: Training labels (num_samples, num_classes) - one-hot encoded
            X_val: Optional validation features
            y_val: Optional validation labels
            verbose: Print training progress

        Returns:
            Training history

        Raises:
            ValueError: If model hasn't been built
        """
        if self.model_wrapper is None:
            raise ValueError("Model not built. Call build_model() first.")

        if verbose:
            print("\n" + "=" * 70)
            print("STARTING TRAINING")
            print("=" * 70 + "\n")
            print(f"Training data:")
            print(f"  X_train shape: {X_train.shape}")
            print(f"  y_train shape: {y_train.shape}")

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            if verbose:
                print(f"\nValidation data:")
                print(f"  X_val shape: {X_val.shape}")
                print(f"  y_val shape: {y_val.shape}")

        # Setup callbacks
        if verbose:
            print("\nSetting up callbacks:")

        callbacks = self._setup_callbacks()

        if verbose:
            print(f"\nTraining configuration:")
            print(f"  Epochs: {self.config.hyperparameters.epochs}")
            print(f"  Batch size: {self.config.hyperparameters.batch_size}")
            print("\n" + "=" * 70)
            print()

        # Train model
        self.history = self.model_wrapper.get_model().fit(
            X_train,
            y_train,
            validation_data=validation_data,
            validation_split=self.config.hyperparameters.validation_split if validation_data is None else 0.0,
            epochs=self.config.hyperparameters.epochs,
            batch_size=self.config.hyperparameters.batch_size,
            class_weight=self.config.hyperparameters.class_weights,
            shuffle=self.config.hyperparameters.shuffle,
            callbacks=callbacks,
            verbose=self.config.hyperparameters.verbose
        )

        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED")
            print("=" * 70)
            self._print_training_summary()

        return self.history

    def _print_training_summary(self):
        """Print summary of training results."""
        if self.history is None:
            return

        history_dict = self.history.history

        print("\nFinal epoch metrics:")

        # Training metrics
        if 'loss' in history_dict:
            print(f"  Training loss:     {history_dict['loss'][-1]:.4f}")
        if 'accuracy' in history_dict:
            print(f"  Training accuracy: {history_dict['accuracy'][-1]:.4f}")

        # Validation metrics
        if 'val_loss' in history_dict:
            print(f"  Validation loss:     {history_dict['val_loss'][-1]:.4f}")
        if 'val_accuracy' in history_dict:
            print(f"  Validation accuracy: {history_dict['val_accuracy'][-1]:.4f}")

        # Best metrics
        print("\nBest metrics:")
        if 'val_accuracy' in history_dict:
            best_val_acc = max(history_dict['val_accuracy'])
            best_epoch = history_dict['val_accuracy'].index(best_val_acc) + 1
            print(f"  Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        elif 'accuracy' in history_dict:
            best_acc = max(history_dict['accuracy'])
            best_epoch = history_dict['accuracy'].index(best_acc) + 1
            print(f"  Best training accuracy: {best_acc:.4f} (epoch {best_epoch})")

        print(f"\nTotal epochs trained: {len(history_dict.get('loss', []))}")

    def save_model(self, filepath: Optional[str] = None):
        """
        Save the trained model.

        Args:
            filepath: Optional path to save model. If None, uses checkpoint path from config.
        """
        if self.model_wrapper is None:
            raise ValueError("No model to save. Build and train a model first.")

        if filepath is None:
            filepath = self.config.callbacks.checkpoint_filepath

        self.model_wrapper.save(filepath, save_config=True)

    def save_history(self, filepath: str = "output/training_history.json"):
        """
        Save training history to JSON file.

        Args:
            filepath: Path to save history JSON
        """
        if self.history is None:
            raise ValueError("No training history available. Train a model first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert history to JSON-serializable format
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]

        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"✓ Training history saved to {filepath}")

    def get_model(self) -> Optional[KerasLSTMModel]:
        """Get the model wrapper."""
        return self.model_wrapper
