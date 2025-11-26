import json
from pathlib import Path

def save_metadata(metadata, output_path):
    """Saves metadata to a specified JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return output_path

def load_metadata(metadata_path):
    """Loads metadata from a specified JSON file."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, 'r') as f:
        return json.load(f)
        
def get_unique_classes_counts(metadata):
    unique_classes = set()
    count_options = {}

    for item in metadata:
        # It's possible for a file to have no detections
        if 'detections' not in item:
            continue
        for cls in item['detections']:
            unique_classes.add(cls['class'])
            if cls['class'] not in count_options:
                count_options[cls['class']] = set()
            count_options[cls['class']].add(cls['count'])

    unique_classes = sorted(list(unique_classes))
    for cls in count_options:
        count_options[cls] = sorted(list(count_options[cls]))
    
    return unique_classes, count_options
