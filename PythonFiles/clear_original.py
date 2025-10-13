import os

# Root folder containing your labeled folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet", "Original")
ORIGINAL_DIR = os.path.abspath(ORIGINAL_DIR)

def clear_original_directory():
    """Clear all files in the Original directory structure"""
    if not os.path.exists(ORIGINAL_DIR):
        print(f"Directory does not exist: {ORIGINAL_DIR}")
        return
    
    total_deleted = 0
    
    for label_folder in os.listdir(ORIGINAL_DIR):
        label_path = os.path.join(ORIGINAL_DIR, label_folder)

        if os.path.isdir(label_path):
            # Delete all files inside the folder
            file_count = 0
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    file_count += 1
                    total_deleted += 1
            print(f"Cleared {file_count} files from: {label_folder}")
        else:
            print(f"Skipped (not a directory): {label_path}")
    
    print(f"\n{'='*60}")
    print(f"Total files deleted: {total_deleted}")
    print(f"Original directory cleared: {ORIGINAL_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Clearing Original directories...\n")
    clear_original_directory()
    print("\nAll Original folders cleared!")