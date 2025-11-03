"""
Utility to migrate Spanish Label Studio annotations to English ActivityLabel enum values.
"""

import json
from typing import Dict, List, Any
from labels_reader.domain.video_labels import ActivityLabel


class LabelMigrator:
    """
    Migrates Spanish Label Studio annotations to English ActivityLabel enum values.
    """

    # Mapping from Spanish labels to English ActivityLabel enum values
    LABEL_MAPPING: Dict[str, str] = {
        "Caminar alejandose (espaldas)": ActivityLabel.WALKING_AWAY_FROM_CAMERA.value,
        "Parado sin movimiento": ActivityLabel.STANDING_STILL.value,
        "Giro 180 izquierda": ActivityLabel.TURNING.value,
        "Giro 180 derecha": ActivityLabel.TURNING.value,
        "Sentarse": ActivityLabel.SITTING_DOWN.value,
        "Sentado sin movimiento": ActivityLabel.SITTING_STILL.value,
        "Ponerse de pie": ActivityLabel.STANDING_UP.value,
        "Caminar acercandose": ActivityLabel.WALKING_TOWARDS_CAMERA.value,
        "Inclinarse izquierda": ActivityLabel.STANDING_STILL.value,
        "Inclinarse derecha": ActivityLabel.STANDING_STILL.value,
    }

    # Labels to discard completely
    DISCARD_LABELS = {"Sentadillas"}

    def migrate_labels(self, input_json_path: str, output_json_path: str) -> None:
        """
        Migrate Spanish labels in Label Studio JSON to English ActivityLabel values.

        Args:
            input_json_path: Path to the input JSON file with Spanish labels
            output_json_path: Path to save the migrated JSON file

        Raises:
            ValueError: If an unmapped label is encountered
            FileNotFoundError: If input JSON file doesn't exist
            json.JSONDecodeError: If input JSON is invalid
        """
        # Load the input JSON
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Migrate the labels
        migrated_data = self._migrate_data(data)

        # Save the migrated JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(migrated_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully migrated labels from {input_json_path} to {output_json_path}")

    def _migrate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Migrate all annotations in the data structure.

        Args:
            data: Label Studio JSON data structure

        Returns:
            Migrated data structure with English labels
        """
        migrated_data = []

        for item in data:
            migrated_item = item.copy()

            # Process annotations if they exist
            if 'annotations' in migrated_item:
                migrated_annotations = []

                for annotation in migrated_item['annotations']:
                    migrated_annotation = annotation.copy()

                    # Process results within each annotation
                    if 'result' in migrated_annotation:
                        migrated_result = self._migrate_timeline_labels(
                            migrated_annotation['result']
                        )
                        migrated_annotation['result'] = migrated_result

                    migrated_annotations.append(migrated_annotation)

                migrated_item['annotations'] = migrated_annotations

            migrated_data.append(migrated_item)

        return migrated_data

    def _migrate_timeline_labels(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Migrate timeline labels within annotation results.

        Args:
            results: List of annotation results

        Returns:
            Migrated results with discarded and mapped labels
        """
        migrated_results = []

        for result in results:
            # Check if this is a timeline labels result
            if result.get('type') == 'timelinelabels' and 'value' in result:
                original_labels = result['value'].get('timelinelabels', [])

                # Filter out discarded labels and migrate the rest
                migrated_labels = []
                for label in original_labels:
                    if label in self.DISCARD_LABELS:
                        # Skip this label entirely
                        continue
                    elif label in self.LABEL_MAPPING:
                        # Map to English label
                        migrated_labels.append(self.LABEL_MAPPING[label])
                    else:
                        # Raise error for unmapped labels
                        raise ValueError(
                            f"Unmapped label encountered: '{label}'. "
                            f"Please add this label to LABEL_MAPPING or DISCARD_LABELS."
                        )

                # Only include the result if there are labels remaining after filtering
                if migrated_labels:
                    migrated_result = result.copy()
                    migrated_result['value'] = result['value'].copy()
                    migrated_result['value']['timelinelabels'] = migrated_labels
                    migrated_results.append(migrated_result)
            else:
                # Keep non-timeline results as-is
                migrated_results.append(result)

        return migrated_results


def main():
    """
    Example usage of the LabelMigrator.
    """
    migrator = LabelMigrator()

    # Example: migrate labels from input to output
    input_path = "data/source2/labels_aux.json"
    output_path = "data/source2/migration.json" 

    try:
        migrator.migrate_labels(input_path, output_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_path}")


if __name__ == "__main__":
    main()
