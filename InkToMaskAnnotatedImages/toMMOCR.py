import json

# Load your input JSON file
input_file = 'bounding_boxes_annotations.json'
output_file = 'mmocr_annotations.json'

with open(input_file, 'r') as f:
    data = json.load(f)

# Initialize the MMOCR format
mmocr_data = {
    "metainfo": {
        "dataset_type": "TextDetDataset",
        "task_name": "textdet",
        "category": [{"id": 0, "name": "text"}]
    },
    "data_list": []
}

# Loop through each image in the input data
for img_path, annotations in data.items():
    img_info = {
        "img_path": img_path,
        "height": None,  # Fill manually if you have height info
        "width": None,   # Fill manually if you have width info
        "instances": []
    }
    
    for anno in annotations:
        x, y, w, h = anno['x'], anno['y'], anno['width'], anno['height']
        instance = {
            "bbox": [x, y, x+w, y+h],
            "bbox_label": 0,
            "polygon": [x, y, x+w, y, x+w, y+h, x, y+h],
            "text": "",  # Text not provided in your input
            "ignore": False
        }
        img_info['instances'].append(instance)
    
    mmocr_data['data_list'].append(img_info)

# Save the output JSON in MMOCR format
with open(output_file, 'w') as f:
    json.dump(mmocr_data, f, indent=4)

print(f"✅ Conversion complete! Saved as {output_file}")
