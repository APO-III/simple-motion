"""
Example usage of MotionOrchestrator to process videos with activity labels.

This example demonstrates how to:
1. Load video annotations from Label Studio JSON
2. Process multiple videos and extract motion features
3. Match extracted features with activity labels
4. Access the labeled features data
5. Export data to CSV format
"""

from infrastructure.containers import Container
from orchestrator.utils.csv_exporter import CSVExporter
import json


def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "PROCESSING VIDEOS WITH LABELS")
    print("=" * 70 + "\n")

    # Initialize dependency injection container
    container = Container()

    # Get orchestrator service
    orchestrator = container.motion_orchestrator()

    # Get target FPS from config
    target_fps = container.config.target_fps()

    # Paths
    videos_dir = "data/source2/combined"
    labels_json = "data/source2/migration.json"

    print(f"Configuration:")
    print(f"  Videos directory: {videos_dir}")
    print(f"  Labels JSON: {labels_json}")
    print(f"  Target FPS: {target_fps}")
    print(f"  Visualizations: No")
    print()

    # Process all videos with labels
    print("Starting video processing...\n")

    labeled_features_list = orchestrator.process_videos_with_labels(
        labels_json_path=labels_json,
        videos_dir=videos_dir,
        target_fps=target_fps,
        generate_visualizations=False  # Set to True to generate frame visualizations
    )

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70 + "\n")

    # Count labels
    label_counts = {}
    videos_processed = set()

    for labeled_feature in labeled_features_list:
        videos_processed.add(labeled_feature.video_id)

        if labeled_feature.has_label():
            label_name = labeled_feature.activity_label.value
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    print(f"Total frames processed: {len(labeled_features_list)}")
    print(f"Videos processed: {len(videos_processed)}")
    print(f"Labeled frames: {sum(label_counts.values())}")
    print(f"Unlabeled frames: {len(labeled_features_list) - sum(label_counts.values())}")
    print()

    # Show activity distribution
    print("Activity Label Distribution:")
    for label_name in sorted(label_counts.keys()):
        count = label_counts[label_name]
        percentage = (count / len(labeled_features_list)) * 100
        print(f"  {label_name:30s}: {count:5d} frames ({percentage:5.2f}%)")
    print()

    # Show sample data from first video
    if labeled_features_list:
        print("Sample data from first video:")
        first_video_id = list(videos_processed)[0]
        video_samples = [lf for lf in labeled_features_list if lf.video_id == first_video_id][:5]

        for labeled_feature in video_samples:
            print(f"\n  Frame {labeled_feature.frame_number}:")
            print(f"    Video: {labeled_feature.video_id}")

            if labeled_feature.has_label():
                print(f"    Activity: {labeled_feature.activity_label.value}")
            else:
                print(f"    Activity: (unlabeled)")

            features = labeled_feature.motion_features
            print(f"    Leg Length: {features.normalized_leg_length:.3f}")
            print(f"    Hip Angle: {features.average_hip_angle:.1f}°")
            print(f"    Knee Angle: {features.average_knee_angle:.1f}°")
            print(f"    Shoulder Vec: ({features.shoulder_vector_x:.3f}, {features.shoulder_vector_z:.3f})")
            print(f"    Ankle Vec: ({features.ankle_vector_x:.3f}, {features.ankle_vector_z:.3f})")

    # Export to CSV
    print("\n" + "=" * 70)
    print("EXPORTING DATA TO CSV")
    print("=" * 70 + "\n")

    exporter = CSVExporter()

    # Option 1: Export all data to a single CSV
    single_csv_path = "output/labeled_features_dataset.csv"
    exporter.export_to_csv(
        labeled_features_list=labeled_features_list,
        output_path=single_csv_path,
        include_unlabeled=True  # Set to False to exclude unlabeled frames
    )

    # Option 2: Export each video to separate CSV files
    # Uncomment to enable:
    # videos_csv_dir = "output/videos_csv"
    # exporter.export_by_video(
    #     labeled_features_list=labeled_features_list,
    #     output_dir=videos_csv_dir,
    #     include_unlabeled=True
    # )

    # Print dataset statistics
    exporter.print_statistics(labeled_features_list)

    # Optional: Also export to JSON
    export_json = False  # Set to True to export JSON as well
    if export_json:
        json_output_file = "output/labeled_features_dataset.json"

        # Convert to dict for JSON serialization
        data_to_export = [lf.to_dict() for lf in labeled_features_list]

        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_export, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported {len(data_to_export)} labeled features to {json_output_file}\n")

    print("=" * 70)
    print("Processing completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
