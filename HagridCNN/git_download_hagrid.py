import zipfile
import os
import shutil

DATA_DIR = 'hagrid_data'

def extract_zip_file(zip_name):
    """Extract a zip file if it exists"""
    zip_path = os.path.join(DATA_DIR, zip_name)
    
    if os.path.exists(zip_path):
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"✓ Extracted {zip_name}")
        return True
    return False

print("Checking for Kaggle downloads...")

# Try different possible zip file names
zip_files = [
    'hagrid-sample-30k-384p.zip',
    'hagrid.zip',
]

extracted = False
for zip_name in zip_files:
    if extract_zip_file(zip_name):
        extracted = True
        break

if not extracted:
    print("✗ No zip files found. Check if already extracted.")

# Organize into expected structure
os.makedirs(os.path.join(DATA_DIR, 'raw_images'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'raw_annotations'), exist_ok=True)

print("\nOrganizing files...")

# Look inside the extracted folder
extracted_main_folder = os.path.join(DATA_DIR, 'hagrid-sample-30k-384p')

if os.path.exists(extracted_main_folder):
    print(f"Found extracted folder: hagrid-sample-30k-384p")
    
    # List what's inside
    inner_folders = os.listdir(extracted_main_folder)
    print(f"Contents: {inner_folders}")
    
    # Process each subfolder
    for folder in inner_folders:
        folder_path = os.path.join(extracted_main_folder, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Check if this folder itself has images
        sample_files = os.listdir(folder_path)[:20]
        has_images = any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in sample_files)
        has_json = any(f.lower().endswith('.json') for f in sample_files)
        
        print(f"  Checking {folder}: images={has_images}, json={has_json}")
        
        if has_json:
            target = os.path.join(DATA_DIR, 'raw_annotations', 'ann_subsample')
            if not os.path.exists(target):
                print(f"    Moving {folder} -> raw_annotations/ann_subsample")
                shutil.move(folder_path, target)
            else:
                print(f"    Target already exists, skipping")
        elif has_images:
            target = os.path.join(DATA_DIR, 'raw_images', 'subsample')
            if not os.path.exists(target):
                print(f"    Moving {folder} -> raw_images/subsample")
                shutil.move(folder_path, target)
            else:
                print(f"    Target already exists, skipping")
        else:
            # Check if folder contains subfolders with images (nested structure)
            print(f"    Checking subfolders inside {folder}...")
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            
            if subfolders:
                print(f"    Found subfolders: {subfolders[:5]}...")
                # Check first subfolder for images
                first_subfolder = os.path.join(folder_path, subfolders[0])
                subfolder_files = os.listdir(first_subfolder)[:10]
                has_nested_images = any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in subfolder_files)
                
                if has_nested_images:
                    print(f"    Found images in subfolders (gesture classes)")
                    target = os.path.join(DATA_DIR, 'raw_images', 'subsample')
                    if not os.path.exists(target):
                        print(f"    Moving {folder} -> raw_images/subsample")
                        shutil.move(folder_path, target)
                    else:
                        print(f"    Target already exists, skipping")
    
    # Remove the now-empty extracted folder
    try:
        if os.path.exists(extracted_main_folder) and not os.listdir(extracted_main_folder):
            os.rmdir(extracted_main_folder)
            print("✓ Cleaned up empty folder")
    except Exception as e:
        print(f"Note: Could not remove folder: {e}")
else:
    # Fallback: look for folders directly in DATA_DIR
    subdirs = [d for d in os.listdir(DATA_DIR) 
               if os.path.isdir(os.path.join(DATA_DIR, d)) 
               and d not in ['raw_images', 'raw_annotations', 'processed_images']]
    
    print(f"Found folders: {subdirs}")
    
    for folder in subdirs:
        folder_path = os.path.join(DATA_DIR, folder)
        sample_files = os.listdir(folder_path)[:20]
        
        has_images = any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in sample_files)
        has_json = any(f.lower().endswith('.json') for f in sample_files)
        
        if has_images:
            target = os.path.join(DATA_DIR, 'raw_images', 'subsample')
            if not os.path.exists(target):
                shutil.move(folder_path, target)
                print(f"✓ Moved {folder} -> raw_images/subsample")
        elif has_json:
            target = os.path.join(DATA_DIR, 'raw_annotations', 'ann_subsample')
            if not os.path.exists(target):
                shutil.move(folder_path, target)
                print(f"✓ Moved {folder} -> raw_annotations/ann_subsample")

# Verify setup
images_folder = os.path.join(DATA_DIR, 'raw_images', 'subsample')
annotations_folder = os.path.join(DATA_DIR, 'raw_annotations', 'ann_subsample')

print("\n" + "="*80)
if os.path.exists(images_folder):
    # Count total images (might be in subfolders)
    image_count = 0
    for root, dirs, files in os.walk(images_folder):
        image_count += len([f for f in files if f.endswith(('.jpg', '.png'))])
    print(f"✓ Images: {image_count} files in {images_folder}")
else:
    print("✗ Images folder not found")

if os.path.exists(annotations_folder):
    json_count = len([f for f in os.listdir(annotations_folder) if f.endswith('.json')])
    print(f"✓ Annotations: {json_count} files in {annotations_folder}")
else:
    print("✗ Annotations folder not found")

print("="*80)

if os.path.exists(images_folder) and os.path.exists(annotations_folder):
    print("\n✓✓✓ Setup complete! Now run: python download_and_setup.py")
else:
    print("\n⚠ Setup incomplete. Check folder structure manually.")