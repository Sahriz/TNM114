import os

# Root folder containing your labeled folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROPPED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet", "Cropped")
CROPPED_DIR = os.path.abspath(CROPPED_DIR)

PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet", "Processed")
PROCESSED_DIR = os.path.abspath(PROCESSED_DIR)

def clear_directory(root_dir, dir_name):
    """Clear all files in a directory structure"""
    if not os.path.exists(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return
    
    for label_folder in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_folder)

        if os.path.isdir(label_path):
            # Delete all files inside the folder
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            print(f"Cleared folder: {label_path}")
        else:
            print(f"Skipped (not a directory): {label_path}")

if __name__ == "__main__":
    print("Clearing Cropped directories...")
    clear_directory(CROPPED_DIR, "Cropped")
    
    print("\nClearing Processed directories...")
    clear_directory(PROCESSED_DIR, "Processed")
    
    print("\nAll Cropped and Processed folders cleared!")