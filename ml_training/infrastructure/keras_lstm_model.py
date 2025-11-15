"""
Keras/TensorFlow LSTM model implementation.

This module provides the concrete implementation of the LSTM architecture
using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import json
from typing import Optional

from ml_training.domain.training_config import LSTMArchitectureConfig


class KerasLSTMModel:
    """
    LSTM model implementation using Keras.

    This is a bidirectional LSTM with dropout for sequence classification
    of motion activities.
    """

    def __init__(self, config: LSTMArchitectureConfig):
        """
        Initialize LSTM model.

        Args:
            config: Architecture configuration
        """
        self.config = config
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """
        Build the LSTM model architecture.

        Architecture:
        1. Input: (timesteps, features)
        2. LSTM Layer 1: Bidirectional, return sequences
        3. Dropout
        4. LSTM Layer 2: Bidirectional, return final state
        5. Dropout
        6. Dense Layer: ReLU activation
        7. Dropout
        8. Output Layer: Softmax for classification

        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name="LSTM_MotionClassifier")

        # Input layer (implicit through first LSTM layer)
        # Shape: (batch_size, timesteps, features)

        # First LSTM layer - capture short-term patterns
        model.add(layers.LSTM(
            units=self.config.lstm1_units,
            return_sequences=True,  # Pass sequences to next LSTM
            recurrent_dropout=self.config.recurrent_dropout,
            input_shape=self.config.input_shape,
            name="lstm_layer_1"
        ))
        model.add(layers.Dropout(
            rate=self.config.dropout_lstm,
            name="dropout_lstm_1"
        ))

        # Second LSTM layer - capture long-term patterns
        model.add(layers.LSTM(
            units=self.config.lstm2_units,
            return_sequences=False,  # Only return final state
            recurrent_dropout=self.config.recurrent_dropout,
            name="lstm_layer_2"
        ))
        model.add(layers.Dropout(
            rate=self.config.dropout_lstm,
            name="dropout_lstm_2"
        ))

        # Dense layer - high-level feature extraction
        model.add(layers.Dense(
            units=self.config.dense_units,
            activation='relu',
            name="dense_layer"
        ))
        model.add(layers.Dropout(
            rate=self.config.dropout_dense,
            name="dropout_dense"
        ))

        # Output layer - classification
        model.add(layers.Dense(
            units=self.config.num_classes,
            activation='softmax',
            name="output_layer"
        ))

        return model

    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.F1Score(name='f1_score', average='weighted')
            ]
        )

    def get_model(self) -> keras.Model:
        """Get the Keras model instance."""
        return self.model

    def summary(self):
        """Print model architecture summary."""
        return self.model.summary()

    def count_parameters(self) -> dict:
        """
        Count trainable and non-trainable parameters.

        Returns:
            Dictionary with parameter counts
        """
        trainable = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])

        return {
            "trainable": int(trainable),
            "non_trainable": int(non_trainable),
            "total": int(trainable + non_trainable)
        }

    def save(self, filepath: str, save_config: bool = True):
        """
        Save model to file.

        Args:
            filepath: Path to save model (.keras format)
            save_config: Whether to save configuration alongside model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")

        # Save configuration
        if save_config:
            config_path = filepath.parent / f"{filepath.stem}_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            print(f"✓ Configuration saved to {config_path}")

    @classmethod
    def load(cls, filepath: str, config: Optional[LSTMArchitectureConfig] = None) -> 'KerasLSTMModel':
        """
        Load model from file.

        Args:
            filepath: Path to model file
            config: Optional configuration (will try to load from file if not provided)

        Returns:
            KerasLSTMModel instance with loaded weights
        """
        filepath = Path(filepath)

        # Load configuration if not provided
        if config is None:
            config_path = filepath.parent / f"{filepath.stem}_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = LSTMArchitectureConfig(**config_dict)
                print(f"✓ Configuration loaded from {config_path}")
            else:
                raise ValueError(f"No configuration provided and no config file found at {config_path}")

        # Create instance
        instance = cls(config)

        # Load weights
        instance.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")

        return instance

    def print_architecture_summary(self):
        """Print detailed architecture summary."""
        print("\n" + "=" * 70)
        print("LSTM MODEL ARCHITECTURE SUMMARY")
        print("=" * 70)

        print("\nConfiguration:")
        print(f"  Input shape: {self.config.input_shape}")
        print(f"  Output classes: {self.config.num_classes}")
        print()

        print("Layer details:")
        print(f"  1. LSTM-1:    {self.config.lstm1_units} units (bidirectional, return_sequences=True)")
        print(f"     Dropout:   {self.config.dropout_lstm}")
        print(f"  2. LSTM-2:    {self.config.lstm2_units} units (bidirectional, return_sequences=False)")
        print(f"     Dropout:   {self.config.dropout_lstm}")
        print(f"  3. Dense:     {self.config.dense_units} units (ReLU)")
        print(f"     Dropout:   {self.config.dropout_dense}")
        print(f"  4. Output:    {self.config.num_classes} units (Softmax)")
        print()

        params = self.count_parameters()
        print("Parameters:")
        print(f"  Trainable:     {params['trainable']:,}")
        print(f"  Non-trainable: {params['non_trainable']:,}")
        print(f"  Total:         {params['total']:,}")

        print("\n" + "=" * 70)
        print("Keras Model Summary:")
        print("=" * 70)
        self.summary()
