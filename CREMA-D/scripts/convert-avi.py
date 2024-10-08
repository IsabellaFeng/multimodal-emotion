import os
import subprocess

def convert_flv_to_avi(source_dir, destination_dir):
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".flv"):
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, filename.replace(".flv", ".avi"))
            
            # Convert the .flv file to .avi using ffmpeg
            command = 'ffmpeg -i "{}" "{}"'.format(source_file, destination_file)

            subprocess.run(command, shell=True)


# Example usage
source_dir = './videos'  
destination_dir = './avi' 

convert_flv_to_avi(source_dir, destination_dir)
