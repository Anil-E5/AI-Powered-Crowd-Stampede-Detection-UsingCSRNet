import os

dataset_dir = "dataset"

for root, dirs, files in os.walk(dataset_dir):
    print("Folder:", root)
    print("Subfolders:", dirs)
    print("Files:", files[:5], "...")  # first 5 files
    print("-"*50)
