"""
Training configuration for LSTM models.

This module defines configuration parameters for model architecture,
training hyperparameters, and callbacks.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LSTMArchitectureConfig:
    """
    Configuration for LSTM model architecture.

    Attributes:
        lstm1_units: Number of units in first LSTM layer
        lstm2_units: Number of units in second LSTM layer
        dense_units: Number of units in dense layer
        dropout_lstm: Dropout rate after LSTM layers
        dropout_dense: Dropout rate after dense layer
        recurrent_dropout: Dropout rate for recurrent connections
        input_shape: Shape of input sequences (timesteps, features)
        num_classes: Number of output classes
    """
    lstm1_units: int = 128
    lstm2_units: int = 64
    dense_units: int = 32
    dropout_lstm: float = 0.3
    dropout_dense: float = 0.2
    recurrent_dropout: float = 0.2
    input_shape: tuple = (30, 7)  # (timesteps, features)
    num_classes: int = 7

    def __post_init__(self):
        """Validate configuration."""
        if self.lstm1_units <= 0 or self.lstm2_units <= 0 or self.dense_units <= 0:
            raise ValueError("All unit counts must be positive")

        if not 0 <= self.dropout_lstm <= 1:
            raise ValueError("dropout_lstm must be between 0 and 1")

        if not 0 <= self.dropout_dense <= 1:
            raise ValueError("dropout_dense must be between 0 and 1")

        if not 0 <= self.recurrent_dropout <= 1:
            raise ValueError("recurrent_dropout must be between 0 and 1")

        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "lstm1_units": self.lstm1_units,
            "lstm2_units": self.lstm2_units,
            "dense_units": self.dense_units,
            "dropout_lstm": self.dropout_lstm,
            "dropout_dense": self.dropout_dense,
            "recurrent_dropout": self.recurrent_dropout,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes
        }


@dataclass
class TrainingHyperparameters:
    """
    Hyperparameters for model training.

    Attributes:
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        validation_split: Fraction of training data to use for validation (if no val set provided)
        class_weights: Optional dictionary of class weights for imbalanced datasets
        shuffle: Whether to shuffle training data
        verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
    """
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.0
    class_weights: Optional[dict] = None
    shuffle: bool = True
    verbose: int = 1

    def __post_init__(self):
        """Validate hyperparameters."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "class_weights": self.class_weights,
            "shuffle": self.shuffle,
            "verbose": self.verbose
        }


@dataclass
class CallbacksConfig:
    """
    Configuration for training callbacks.

    Attributes:
        use_early_stopping: Enable early stopping
        early_stopping_patience: Epochs with no improvement before stopping
        early_stopping_monitor: Metric to monitor for early stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        restore_best_weights: Restore weights from best epoch

        use_reduce_lr: Enable learning rate reduction on plateau
        reduce_lr_patience: Epochs with no improvement before reducing LR
        reduce_lr_factor: Factor by which to reduce learning rate
        reduce_lr_min_lr: Minimum learning rate

        use_model_checkpoint: Save model checkpoints
        checkpoint_filepath: Path to save model checkpoints
        checkpoint_save_best_only: Only save when model improves

        use_tensorboard: Enable TensorBoard logging
        tensorboard_log_dir: Directory for TensorBoard logs
    """
    use_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 0.001
    restore_best_weights: bool = True

    use_reduce_lr: bool = True
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    reduce_lr_min_lr: float = 1e-7

    use_model_checkpoint: bool = True
    checkpoint_filepath: str = "output/models/lstm_best_model.keras"
    checkpoint_save_best_only: bool = True

    use_tensorboard: bool = True
    tensorboard_log_dir: str = "output/logs/tensorboard"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "use_early_stopping": self.use_early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "restore_best_weights": self.restore_best_weights,
            "use_reduce_lr": self.use_reduce_lr,
            "reduce_lr_patience": self.reduce_lr_patience,
            "reduce_lr_factor": self.reduce_lr_factor,
            "reduce_lr_min_lr": self.reduce_lr_min_lr,
            "use_model_checkpoint": self.use_model_checkpoint,
            "checkpoint_filepath": self.checkpoint_filepath,
            "checkpoint_save_best_only": self.checkpoint_save_best_only,
            "use_tensorboard": self.use_tensorboard,
            "tensorboard_log_dir": self.tensorboard_log_dir
        }


@dataclass
class TrainingConfig:
    """
    Complete configuration for LSTM training.

    Combines architecture, hyperparameters, and callbacks configuration.
    """
    architecture: LSTMArchitectureConfig = field(default_factory=LSTMArchitectureConfig)
    hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)

    def to_dict(self) -> dict:
        """Convert complete configuration to dictionary."""
        return {
            "architecture": self.architecture.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "callbacks": self.callbacks.to_dict()
        }

    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("LSTM TRAINING CONFIGURATION")
        print("=" * 70)

        print("\nArchitecture:")
        print(f"  Input shape: {self.architecture.input_shape}")
        print(f"  LSTM layers: {self.architecture.lstm1_units} → {self.architecture.lstm2_units}")
        print(f"  Dense layer: {self.architecture.dense_units}")
        print(f"  Output classes: {self.architecture.num_classes}")
        print(f"  Dropout (LSTM): {self.architecture.dropout_lstm}")
        print(f"  Dropout (Dense): {self.architecture.dropout_dense}")
        print(f"  Recurrent dropout: {self.architecture.recurrent_dropout}")

        print("\nHyperparameters:")
        print(f"  Epochs: {self.hyperparameters.epochs}")
        print(f"  Batch size: {self.hyperparameters.batch_size}")
        print(f"  Learning rate: {self.hyperparameters.learning_rate}")
        print(f"  Validation split: {self.hyperparameters.validation_split}")
        print(f"  Class weights: {self.hyperparameters.class_weights is not None}")

        print("\nCallbacks:")
        if self.callbacks.use_early_stopping:
            print(f"  ✓ Early stopping (patience={self.callbacks.early_stopping_patience})")
        if self.callbacks.use_reduce_lr:
            print(f"  ✓ Reduce LR on plateau (patience={self.callbacks.reduce_lr_patience})")
        if self.callbacks.use_model_checkpoint:
            print(f"  ✓ Model checkpoint: {self.callbacks.checkpoint_filepath}")
        if self.callbacks.use_tensorboard:
            print(f"  ✓ TensorBoard: {self.callbacks.tensorboard_log_dir}")

        print("=" * 70 + "\n")
