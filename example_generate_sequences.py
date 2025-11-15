"""
Example script to generate temporal sequences from motion features CSV.

This demonstrates:
1. Loading motion features from CSV
2. Generating sliding window sequences (30, 7)
3. Splitting into train/validation/test sets
4. Analyzing the resulting dataset

Usage:
    python example_generate_sequences.py
"""

from ml_training.domain.sequence import SequenceGeneratorConfig
from ml_training.use_cases.sequence_generator import SequenceGenerator
from ml_training.utils.data_splitter import DataSplitter
from ml_training.utils.label_encoder import LabelEncoder


def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "SEQUENCE GENERATION EXAMPLE")
    print("=" * 80 + "\n")

    # =========================================================================
    # STEP 1: Configure sequence generation
    # =========================================================================
    print("Step 1: Configuring sequence generator")
    print("-" * 80)

    config = SequenceGeneratorConfig(
        window_size=30,    # 30 frames per sequence (1 second @ 30 FPS)
        stride=15,         # 15 frames stride (50% overlap)
        min_segment_length=30  # Skip segments shorter than 30 frames
    )

    print(f"Configuration:")
    print(f"  Window size: {config.window_size} frames")
    print(f"  Stride: {config.stride} frames")
    print(f"  Overlap: {((config.window_size - config.stride) / config.window_size) * 100:.1f}%")
    print(f"  Min segment length: {config.min_segment_length} frames")
    print(f"  Number of features: {config.num_features}")
    print()

    # =========================================================================
    # STEP 2: Generate sequences from CSV
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Generating sequences from CSV")
    print("=" * 80 + "\n")

    generator = SequenceGenerator(config=config)

    # Option A: Single source
    # dataset = generator.generate_from_csv(
    #     csv_path="results/raw/source3.csv",
    #     source_name="source3",
    #     verbose=True
    # )

    # Option B: Multiple sources (recommended)
    dataset = generator.generate_from_multiple_csvs(
        csv_paths=[
            ("results/raw/source1.csv", "source1"),
            ("results/raw/source3.csv", "source3"),
        ],
        verbose=True
    )

    # =========================================================================
    # STEP 3: Print dataset statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Dataset Statistics")
    print("=" * 80)

    dataset.print_statistics()

    # =========================================================================
    # STEP 4: Inspect shapes and data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Data Inspection")
    print("=" * 80 + "\n")

    X = dataset.get_X()
    y = dataset.get_y()

    print(f"X shape: {X.shape}")
    print(f"  - Number of sequences: {X.shape[0]}")
    print(f"  - Timesteps per sequence: {X.shape[1]}")
    print(f"  - Features per timestep: {X.shape[2]}")
    print()

    print(f"y shape: {y.shape}")
    print(f"  - Number of labels: {y.shape[0]}")
    print(f"  - Unique classes: {len(dataset.label_to_index)}")
    print()

    print("Label encoding:")
    for label, idx in sorted(dataset.label_to_index.items(), key=lambda x: x[1]):
        count = sum(1 for seq in dataset.sequences if seq.label == label)
        print(f"  {idx}: {label:30s} ({count} sequences)")
    print()

    # Show sample sequence
    if len(dataset.sequences) > 0:
        sample_seq = dataset.sequences[0]
        print("Sample sequence:")
        print(f"  Video: {sample_seq.video_id}")
        print(f"  Frames: {sample_seq.start_frame} - {sample_seq.end_frame}")
        print(f"  Label: {sample_seq.label}")
        print(f"  Shape: {sample_seq.shape}")
        print(f"  Source: {sample_seq.source}")
        print()
        print(f"  First 3 frames of features:")
        for i in range(min(3, sample_seq.num_timesteps)):
            print(f"    Frame {sample_seq.start_frame + i}: {sample_seq.features[i]}")
        print()

    # =========================================================================
    # STEP 5: Split into train/validation/test sets
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Splitting into Train/Validation/Test Sets")
    print("=" * 80)

    splitter = DataSplitter(random_seed=42)

    train_dataset, val_dataset, test_dataset = splitter.split_by_video(
        dataset=dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by_label=True,
        verbose=True
    )

    # =========================================================================
    # STEP 6: Get arrays ready for model training
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Prepare Data for Model Training")
    print("=" * 80 + "\n")

    X_train = train_dataset.get_X()
    y_train = train_dataset.get_y()

    X_val = val_dataset.get_X()
    y_val = val_dataset.get_y()

    X_test = test_dataset.get_X()
    y_test = test_dataset.get_y()

    print("Training data:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print()

    print("Validation data:")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_val shape: {y_val.shape}")
    print()

    print("Test data:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print()

    # For categorical crossentropy loss, convert to one-hot encoding
    num_classes = len(dataset.label_to_index)
    y_train_cat = train_dataset.get_y_categorical(num_classes)
    y_val_cat = val_dataset.get_y_categorical(num_classes)
    y_test_cat = test_dataset.get_y_categorical(num_classes)

    print(f"One-hot encoded labels (for categorical crossentropy):")
    print(f"  y_train_categorical shape: {y_train_cat.shape}")
    print(f"  y_val_categorical shape: {y_val_cat.shape}")
    print(f"  y_test_categorical shape: {y_test_cat.shape}")
    print()

    # =========================================================================
    # STEP 7: Save label encoder for future use
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Save Label Encoder")
    print("=" * 80 + "\n")

    encoder = LabelEncoder(
        label_to_index=dataset.label_to_index,
        index_to_label=dataset.index_to_label
    )

    encoder.save("output/label_encoder.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Generated {len(dataset)} sequences from CSV")
    print(f"✓ Sequence shape: {X.shape[1:]} (timesteps, features)")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Train: {len(train_dataset)} sequences from {train_dataset.get_statistics()['num_videos']} videos")
    print(f"✓ Val:   {len(val_dataset)} sequences from {val_dataset.get_statistics()['num_videos']} videos")
    print(f"✓ Test:  {len(test_dataset)} sequences from {test_dataset.get_statistics()['num_videos']} videos")
    print()
    print("Next steps:")
    print("  1. Use X_train, y_train_cat for LSTM model training")
    print("  2. Use X_val, y_val_cat for validation during training")
    print("  3. Use X_test, y_test_cat for final model evaluation")
    print("  4. Load label_encoder.json for inference/deployment")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
