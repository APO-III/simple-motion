"""
Example usage of CSVExporter to export labeled motion features to CSV.

This example demonstrates:
1. Processing videos with labels
2. Exporting to single CSV file (all videos combined)
3. Exporting to multiple CSV files (one per video)
4. Viewing dataset statistics
"""

from infrastructure.containers import Container
from orchestrator.utils.csv_exporter import CSVExporter


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "CSV EXPORT EXAMPLE")
    print("=" * 70 + "\n")

    # Step 1: Process videos and get labeled features
    print("Step 1: Processing videos with labels...")
    print("-" * 70)

    container = Container()
    orchestrator = container.motion_orchestrator()
    target_fps = container.config.target_fps()
    exporter = container.exporter()
    
    # Elements: 
    
    data = {
        
            # "source1": {
            #     "videos_dir": "data/source1/combined",
            #     "labels_json": "data/source1/labels.json",
            #     "output" : "output/source1.csv",
            #     "fps": 30.0
            # },
            "source2": {
                "videos_dir": "data/source2/combined",
                "labels_json": "data/source2/labels.json",
                "output" : "output/source2.csv",
                "fps": 30.0
                
            },
            # "source3": {
            #     "videos_dir": "data/source3/combined",
            #     "labels_json": "data/source3/labels.json",
            #     "output" : "output/source3.csv",
            #     "fps": 30.0
            # }
    }
    

   
    for source, params in data.items():
        videos_dir = params["videos_dir"]
        labels_json = params["labels_json"]
        output_path = params["output"]
        fps = params["fps"]
        print(f"\nProcessing {source}:")
        print(f"  Videos directory: {videos_dir}")
        print(f"  Labels JSON: {labels_json}")
        print(f"  Target FPS: {fps}")   
        labeled_features_list = orchestrator.process_videos_with_labels(
            labels_json_path=labels_json,
            videos_dir=videos_dir,
            target_fps=fps,
            generate_visualizations=False
        )
        print(f"\n✓ Processed {len(labeled_features_list)} frames")

        # Step 2: Create CSV exporter
        print("\n" + "=" * 70)
        print("Step 2: Exporting to CSV")
        print("-" * 70 + "\n")

        # Export only labeled frames
        
        exporter.export_to_csv(
            labeled_features_list=labeled_features_list,
            output_path=output_path,
            include_unlabeled=False
        )

        # Step 3: Show statistics
        print("\n" + "=" * 70)
        print("Step 3: Dataset Statistics")
        print("-" * 70)
        exporter.print_statistics(labeled_features_list)

if __name__ == "__main__":
    main()
