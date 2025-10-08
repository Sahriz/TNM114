import os

# Root folder containing your labeled folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "TemporarySet")
ROOT_DIR = os.path.abspath(ROOT_DIR)

def clear_cropped_folders():
    for label_folder in os.listdir(ROOT_DIR):
        label_path = os.path.join(ROOT_DIR, label_folder)
        cropped_path = os.path.join(label_path, "Cropped")

        if os.path.isdir(cropped_path):
            # Delete all files inside Cropped
            for file_name in os.listdir(cropped_path):
                file_path = os.path.join(cropped_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            print(f"Cleared folder: {cropped_path}")
        else:
            print(f"Skipped (no Cropped folder): {cropped_path}")

if __name__ == "__main__":
    clear_cropped_folders()
    print("\nAll Cropped folders cleared!")
