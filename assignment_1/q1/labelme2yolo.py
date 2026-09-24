# Quick script to convert Labelme JSON annotations to YOLO format
# Produced entirely by Gemini, only for conversion of Labelme JSON to YOLO format.

import json
import os
import glob

# --- CONFIGURATION ---
# Replace these with your actual directory paths
INPUT_DIR = './dataset/labels/test' 
OUTPUT_DIR = './dataset/labels/test'

# Define your classes based on your dataset.yaml
CLASS_MAPPING = {
    'box': 0
    # Add other classes here if needed, e.g., 'person': 1
}
# ---------------------

def convert_labelme_to_yolo(input_dir, output_dir, class_mapping):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    if not json_files:
        print(f"No .json files found in {input_dir}")
        return

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {json_file}. Skipping.")
                continue
            
        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')
        
        if not img_width or not img_height:
            print(f"Missing image dimensions in {json_file}. Skipping.")
            continue
        
        # Prepare corresponding .txt output path
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            for shape in data.get('shapes', []):
                label = shape.get('label')
                shape_type = shape.get('shape_type')
                points = shape.get('points')
                
                # Skip unmapped classes or non-rectangle shapes
                if label not in class_mapping or shape_type != 'rectangle':
                    continue
                    
                class_id = class_mapping[label]
                
                # Labelme rectangle points: [[x1, y1], [x2, y2]]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Get absolute min/max to calculate width/height safely
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                
                # Calculate absolute center and dimensions
                abs_w = xmax - xmin
                abs_h = ymax - ymin
                abs_cx = xmin + (abs_w / 2.0)
                abs_cy = ymin + (abs_h / 2.0)
                
                # Normalize values to 0.0 - 1.0 based on image size
                norm_cx = abs_cx / img_width
                norm_cy = abs_cy / img_height
                norm_w = abs_w / img_width
                norm_h = abs_h / img_height
                
                # Ensure coordinates are within valid YOLO range (0 to 1)
                norm_cx = max(0.0, min(1.0, norm_cx))
                norm_cy = max(0.0, min(1.0, norm_cy))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                # Write YOLO format line
                txt_file.write(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
    print(f"Successfully converted {len(json_files)} files to {output_dir}")

if __name__ == '__main__':
    convert_labelme_to_yolo(INPUT_DIR, OUTPUT_DIR, CLASS_MAPPING)
