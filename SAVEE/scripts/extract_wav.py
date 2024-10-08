import os
import subprocess

# Define the directory containing the .avi files
source_dir = 'all_videos' 

# Loop through all .avi files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".avi"):
        # Full path to the source .avi file
        inputfile = os.path.join(source_dir, filename)
        # Full path to the destination .wav file in the same directory
        outputfile = os.path.join(source_dir, os.path.splitext(filename)[0] + '.wav')
        
        # Extract the audio from the .avi file and save it as a .wav file using ffmpeg
        subprocess.run(['ffmpeg', '-i', inputfile, outputfile])
        print(f'Extracted audio from {filename} and saved as {outputfile}')
