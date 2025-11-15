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
    video_path = "data/source2/combined/VIDEO_01.mp4"

    # Without visualizations
    features_map = orchestrator.process_video(
        video_path=video_path,
        target_fps=23.0,    
        video_id="test_video_001",
        generate_visualizations=True
    )
    
    print(f"Processed {len(features_map)} frames")
    
if __name__ == "__main__":
    main()
