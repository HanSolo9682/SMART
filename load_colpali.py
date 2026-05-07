import os
import json
import datasets
from tqdm import tqdm

import sys
dataset_path = sys.argv[1]

print("Loading dataset...")
data = datasets.load_dataset(dataset_path, split="train")

# 2. Define your output paths
output_base_dir = os.path.join(dataset_path, "colpali_extracted_data")
image_dir = os.path.join(output_base_dir, "images")
json_path = os.path.join(output_base_dir, "queries.json")

# Create the output directories if they don't exist
os.makedirs(image_dir, exist_ok=True)

# 3. Initialize a list to hold our JSON metadata
metadata_records = []

print(f"Processing and saving {len(data)} examples...")

# 4. Iterate through the dataset
for i, item in enumerate(tqdm(data, desc="Extracting Data")):
    img = item["image"]
    query = item["query"]
    original_filename = item.get("image_filename", "")
    
    # Generate a unique filename using the index to prevent accidental overwrites
    # (Using .jpg to save space, but you can change this to .png if you prefer lossless)
    new_filename = f"image_{i:06d}.jpg"
    img_save_path = os.path.join(image_dir, new_filename)
    
    # Convert image to RGB before saving as JPEG to avoid errors with RGBA/grayscale formats
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Save the image to disk
    img.save(img_save_path, format="JPEG", quality=95)
    
    # Store the mapping and any other relevant metadata you might want later
    metadata_records.append({
        "saved_filename": new_filename,
        "original_filename": original_filename,
        "query": query,
        "answer": item.get("answer", ""),
        "source": item.get("source", "")
    })

# 5. Save the metadata as a JSON file
print("Saving JSON metadata...")
with open(json_path, "w", encoding="utf-8") as json_file:
    # ensure_ascii=False keeps special characters intact
    json.dump(metadata_records, json_file, indent=4, ensure_ascii=False)

print(f"Done! Images saved to: {image_dir}")
print(f"Done! Queries saved to: {json_path}")