import os

DATA_DIR = 'hagrid_data'
raw_images_folder = os.path.join(DATA_DIR, 'raw_images', 'subsample')

print(f"Checking: {raw_images_folder}")
print(f"Exists: {os.path.exists(raw_images_folder)}")

if os.path.exists(raw_images_folder):
    contents = os.listdir(raw_images_folder)
    print(f"\nContents ({len(contents)} items):")
    print(contents[:20])  # Show first 20 items
    
    # Check if there are subfolders
    subfolders = [item for item in contents if os.path.isdir(os.path.join(raw_images_folder, item))]
    files = [item for item in contents if os.path.isfile(os.path.join(raw_images_folder, item))]
    
    print(f"\nSubfolders: {len(subfolders)}")
    if subfolders:
        print(f"  Examples: {subfolders[:5]}")
        # Check what's inside a subfolder
        first_subfolder = os.path.join(raw_images_folder, subfolders[0])
        subfolder_contents = os.listdir(first_subfolder)[:10]
        print(f"  Inside {subfolders[0]}: {subfolder_contents}")
    
    print(f"\nImage files directly: {len(files)}")
    if files:
        print(f"  Examples: {files[:5]}")
    
    # Check annotation structure
    annotations_folder = os.path.join(DATA_DIR, 'raw_annotations', 'ann_subsample')
    print(f"\n\nChecking annotations: {annotations_folder}")
    
    if os.path.exists(annotations_folder):
        # Load one annotation file to see the structure
        import json
        ann_file = os.path.join(annotations_folder, 'call.json')
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Get first 3 keys
        sample_keys = list(data.keys())[:3]
        print(f"\nSample annotation keys: {sample_keys}")
        print(f"First annotation entry:")
        print(json.dumps(data[sample_keys[0]], indent=2))