import os
import shutil

# Define the base directory containing the folders
base_dir = 'AudioVisualClip'

# Define the destination directory where all renamed files will be saved
destination_dir = 'all_videos'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List all subdirectories in the base directory
subfolders = ['DC', 'JE', 'JK', 'KL']

# Iterate through each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    
    # Check if the subfolder exists
    if os.path.exists(subfolder_path):
        # Iterate through each file in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.avi'):
                # Construct the new filename
                new_filename = f"{subfolder}_{filename}"
                
                # Define the source and destination paths
                source_file = os.path.join(subfolder_path, filename)
                destination_file = os.path.join(destination_dir, new_filename)
                
                # Move and rename the file
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")

print("All files have been renamed and copied successfully.")
