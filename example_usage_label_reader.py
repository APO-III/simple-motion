"""
Example usage of LabelStudioReader to load and work with video annotations.

This example demonstrates:
1. How to migrate Spanish labels to English using LabelMigrator
2. How to load migrated annotations using LabelStudioReader
3. How to access video annotations and frame labels
"""

from labels_reader.utils.migrate_labels import LabelMigrator
from labels_reader.use_cases.label_studio_reader import LabelStudioReader


def example_migrate_labels():
    """
    Example 1: Migrate Spanish labels to English format.
    """
    print("=" * 60)
    print("EXAMPLE 1: Migrating Spanish labels to English")
    print("=" * 60)

    migrator = LabelMigrator()

    input_path = "data/source2/labels_aux.json"
    output_path = "data/source2/migration.json"

    try:
        migrator.migrate_labels(input_path, output_path)
        print(f"✓ Successfully migrated labels")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
    except ValueError as e:
        print(f"✗ Error: {e}")

    print()


def example_load_single_json():
    """
    Example 2: Load annotations from a single JSON file.
    """
    print("=" * 60)
    print("EXAMPLE 2: Loading annotations from JSON")
    print("=" * 60)

    reader = LabelStudioReader()

    # Load labels from migrated JSON
    labels_json_path = "data/source2/migration.json"

    try:
        labels_dataset = reader.read_labels_from_json(labels_json_path)

        print(f"✓ Loaded annotations for {len(labels_dataset.get_all_video_ids())} videos")
        print()

        # Show all video IDs
        print("Video IDs found:")
        for video_id in labels_dataset.get_all_video_ids()[:5]:  # Show first 5
            print(f"  - {video_id}")

        if len(labels_dataset.get_all_video_ids()) > 5:
            print(f"  ... and {len(labels_dataset.get_all_video_ids()) - 5} more")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")

    print()


def example_load_with_video_files():
    """
    Example 3: Load annotations and match with video files from directory.
    """
    print("=" * 60)
    print("EXAMPLE 3: Loading annotations with video file paths")
    print("=" * 60)

    reader = LabelStudioReader()

    videos_dir = "data/source2/combined"
    labels_json_path = "data/source2/migration.json"

    try:
        # Load both labels and video file paths
        labels_dataset, video_files = reader.load_dataset(
            videos_dir=videos_dir,
            labels_json_path=labels_json_path
        )

        print(f"✓ Loaded {len(labels_dataset.get_all_video_ids())} annotations")
        print(f"✓ Found {len(video_files)} video files")
        
        

        # Show matched videos
        print("Matched videos:")
        matched_count = 0
        for video_id in labels_dataset.get_all_video_ids():  
            if video_id in video_files:
                matched_count += 1
                print(f"  ✓ {video_id}")
                print(f"    Path: {video_files[video_id]}")
            else:
                print(f"  ✗ {video_id} (video file not found)")

        print(f"\nTotal matched: {matched_count} videos")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")

    print()


def example_query_frame_labels():
    """
    Example 4: Query labels for specific frames in a video.
    """
    print("=" * 60)
    print("EXAMPLE 4: Querying frame labels")
    print("=" * 60)

    reader = LabelStudioReader()
    labels_json_path = "data/source2/migration.json"

    try:
        labels_dataset = reader.read_labels_from_json(labels_json_path)

        # Get first video
        video_ids = labels_dataset.get_all_video_ids()
        if not video_ids:
            print("No videos found in dataset")
            return

        video_id = video_ids[0]
        video_annotation = labels_dataset.get_video_annotation(video_id)

        print(f"Video: {video_id}")
        print(f"Total frame ranges: {len(video_annotation.frame_ranges)}")
        print()

        # Show all frame ranges
        print("Frame ranges and activities:")
        for frame_range in video_annotation.frame_ranges[:10]:  # Show first 10
            print(f"  Frames {frame_range.start_frame:4d} - {frame_range.end_frame:4d}: {frame_range.activity.value}")

        if len(video_annotation.frame_ranges) > 10:
            print(f"  ... and {len(video_annotation.frame_ranges) - 10} more ranges")
        print()

        # Query specific frames
        print("Querying specific frames:")
        test_frames = [1, 50, 100, 200, 300]
        for frame_num in test_frames:
            activity = video_annotation.get_label_for_frame(frame_num)
            if activity:
                print(f"  Frame {frame_num:3d}: {activity.value}")
            else:
                print(f"  Frame {frame_num:3d}: (unlabeled)")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")

    print()


def example_full_workflow():
    """
    Example 5: Complete workflow - migrate, load, and process.
    """
    print("=" * 60)
    print("EXAMPLE 5: Complete workflow")
    print("=" * 60)

    # Step 1: Migrate labels
    print("Step 1: Migrating labels...")
    migrator = LabelMigrator()

    spanish_json = "data/source2/labels_aux.json"
    english_json = "data/source2/migration.json"

    try:
        migrator.migrate_labels(spanish_json, english_json)
        print("✓ Migration complete")
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return

    # Step 2: Load dataset
    print("\nStep 2: Loading dataset...")
    reader = LabelStudioReader()
    videos_dir = "data/source2/videos"

    try:
        labels_dataset, video_files = reader.load_dataset(
            videos_dir=videos_dir,
            labels_json_path=english_json
        )
        print(f"✓ Loaded {len(labels_dataset.get_all_video_ids())} videos")
    except Exception as e:
        print(f"✗ Loading failed: {e}")
        return

    # Step 3: Process each video
    print("\nStep 3: Processing videos...")
    for video_id in labels_dataset.get_all_video_ids()[:2]:  # Process first 2
        if video_id not in video_files:
            print(f"✗ Video file not found for {video_id}")
            continue

        video_path = video_files[video_id]
        video_annotation = labels_dataset.get_video_annotation(video_id)

        print(f"\n  Video: {video_id}")
        print(f"    Path: {video_path}")
        print(f"    Frame ranges: {len(video_annotation.frame_ranges)}")

        # Count activities
        activity_counts = {}
        for frame_range in video_annotation.frame_ranges:
            activity_name = frame_range.activity.value
            frame_count = frame_range.end_frame - frame_range.start_frame + 1
            activity_counts[activity_name] = activity_counts.get(activity_name, 0) + frame_count

        print(f"    Activity breakdown:")
        for activity, count in sorted(activity_counts.items()):
            print(f"      - {activity}: {count} frames")

    print()


def main():
    """
    Run all examples.

    To run individual examples, comment out the ones you don't want to run.
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "LABEL STUDIO READER - EXAMPLES" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run examples
    # example_migrate_labels()
    #example_load_single_json()
    example_load_with_video_files()
    #example_query_frame_labels()
    # example_full_workflow()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
