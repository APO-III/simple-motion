"""
Example usage of the Motion Analysis System with Label Studio annotations.
"""

from infrastructure.containers import Container


def main():
    # Initialize dependency injection container
    container = Container()

    # Get services
    label_reader = container.label_studio_reader()
    orchestrator = container.motion_orchestrator()

    # Get target FPS from config
    target_fps = container.config.target_fps()

    # Paths
    videos_dir = "data/videos"
    labels_json = "data/labels/annotations.json"

    # Load labels and video files
    print("Loading labels and video files...")
    labels_dataset, video_files = label_reader.load_dataset(
        videos_dir=videos_dir,
        labels_json_path=labels_json
    )

    print(f"Found {len(video_files)} videos")
    print(f"Loaded annotations for {len(labels_dataset.get_all_video_ids())} videos")
    print(f"Using target FPS: {target_fps}")

    # Process each video
    for video_id in labels_dataset.get_all_video_ids():
        if video_id not in video_files:
            print(f"Warning: Video file not found for '{video_id}'")
            continue

        video_path = video_files[video_id]
        video_annotation = labels_dataset.get_video_annotation(video_id)

        print(f"\nProcessing video: {video_id}")
        print(f"  Path: {video_path}")
        print(f"  Annotations: {len(video_annotation.frame_ranges)} frame ranges")

        # Process video and extract features
        features_map = orchestrator.process_video(
            video_path=video_path,
            target_fps=target_fps,
            video_id=video_id,
            generate_visualizations=True,
            output_dir=f"output_validations/{video_id}"
        )

        print(f"  Extracted features for {len(features_map)} frames")

        # Show sample results with labels
        sample_frames = list(features_map.keys())[:5]
        for frame_num in sample_frames:
            motion_features = features_map[frame_num]
            activity = video_annotation.get_label_for_frame(frame_num)
            activity_str = activity.value if activity else "unlabeled"

            print(f"\n  Frame {frame_num}: {activity_str}")
            print(f"    Leg Length: {motion_features.normalized_leg_length:.3f}")
            print(f"    Hip Angle: {motion_features.average_hip_angle:.1f}°")
            print(f"    Knee Angle: {motion_features.average_knee_angle:.1f}°")


if __name__ == "__main__":
    main()
