import json
import shutil
import os

# Paths (Update these paths as needed)
annotation_file = "annotation.json"  # Update with the actual file path
images_directory = "data/train/images"  # Directory where all images are stored
output_directory = "filtered_images"  # Directory to save the selected images

# Load the JSON annotation file
with open(annotation_file, "r") as f:
    data = json.load(f)

# Extract the first 20 image filenames
selected_images = data["images"][:20]
selected_filenames = [img["file_name"] for img in selected_images]

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Copy the selected images to the new directory
for filename in selected_filenames:
    source_path = os.path.join(images_directory, filename)
    destination_path = os.path.join(output_directory, filename)

    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied: {filename}")
    else:
        print(f"Warning: {filename} not found in {images_directory}")

print(f"Selected images saved in {output_directory}")
