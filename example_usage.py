"""
Example usage of the Motion Analysis System.
"""

from infrastructure.containers import Container


def main():
    # Initialize dependency injection container
    container = Container()

    # Get the motion orchestrator
    orchestrator = container.motion_orchestrator()

    # Process video and extract features
    video_path = "data/test.mp4"

    # Without visualizations
    features_map = orchestrator.process_video(
        video_path=video_path,
        target_fps=10.0,
        video_id="test_video_001",
        generate_visualizations=False
    )

    # Print results
    print(f"Processed {len(features_map)} frames")
    for frame_number, motion_features in features_map.items():
        print(f"\nFrame {frame_number}:")
        print(f"  Normalized Leg Length: {motion_features.normalized_leg_length:.4f}")
        print(f"  Hip Angle: {motion_features.average_hip_angle:.2f}°")
        print(f"  Knee Angle: {motion_features.average_knee_angle:.2f}°")

    # With visualizations (optional)
    features_map_with_viz = orchestrator.process_video(
        video_path=video_path,
        target_fps=10.0,
        video_id="test_video_002",
        generate_visualizations=True,
        output_dir="output_validations"
    )

    print(f"\nGenerated {len(features_map_with_viz)} visualization images in output_validations/")


if __name__ == "__main__":
    main()
