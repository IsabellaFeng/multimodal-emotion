import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import shutil
import cv2
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

# Define a dictionary to map file label codes to numerical emotion classes
label_map = {
    'a': 0,  # anger
    'd': 1,  # disgust
    'f': 2,  # fear
    'h': 3,  # happiness
    'n': 4,  # neutral
    'sa': 5, # sadness
    'su': 6  # surprise
}

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def max_frames_in_directory(directory_path):
    max_frames = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.avi'):
                full_path = os.path.join(root, file)
                frames = get_frame_count(full_path)
                if frames > max_frames:
                    max_frames = frames
    return max_frames

def pad_symmetric(data, max_length):
    pad_total = max_length - data.shape[0]
    pad_begin = pad_total // 2
    pad_end = pad_total - pad_begin
    return np.pad(data, ((pad_begin, pad_end), (0, 0)), 'constant')

def pad_audio_symmetric(feature, max_length):
    pad_total = max_length - feature.shape[1]
    pad_begin = pad_total // 2
    pad_end = pad_total - pad_begin
    return np.pad(feature, ((0, 0), (pad_begin, pad_end)), 'constant')

    
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    frame_rate = 59.8  # in frames per second
    hop_length = int(sr / frame_rate)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10, hop_length=hop_length)

    # Load Wav2Vec2 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
    
    # Process the audio for Wav2Vec2
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    input_values = processor(y_resampled, return_tensors="pt", sampling_rate=16000).input_values
    
    # Extract Wav2Vec2 features
    with torch.no_grad():
        wav2vec_features = model(input_values).last_hidden_state.squeeze(0).numpy()

    # Resample Wav2Vec2 features to match the MFCC time dimension
    wav2vec_features_resampled = librosa.resample(wav2vec_features.T, orig_sr=wav2vec_features.shape[0], target_sr=mfcc.shape[1]).T
    
    # Ensure the time dimensions match
    if wav2vec_features_resampled.shape[0] != mfcc.shape[1]:
        min_length = min(mfcc.shape[1], wav2vec_features_resampled.shape[0])
        mfcc = mfcc[:, :min_length]
        wav2vec_features_resampled = wav2vec_features_resampled[:min_length, :]

    # Concatenate MFCC and Wav2Vec2 features along the feature dimension (axis 0)
    combined_features = np.concatenate((mfcc, wav2vec_features_resampled.T), axis=0)
    
    return combined_features

def extract_visual_features_openface(video_path, processed_output):
    os.system(f"../multimodal-emotion-recognition/opencv-4.1.0/build/OpenFace/build/bin/FeatureExtraction -f {video_path} -out_dir {processed_output}")
    csv_path = os.path.join(processed_output, os.path.splitext(os.path.basename(video_path))[0] + ".csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        au_features = df.filter(regex='^AU').values
        print(f"Extracted features for {video_path}: {au_features.shape}")
        # Delete the file after reading its contents
        os.remove(csv_path)
        return au_features
    else:
        print(f"Feature extraction failed for {video_path}")
        return None

def extract_efficientface_features(video_path, num_frames):
    # Load EfficientNet model (EfficientFace backbone)
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.eval()  # Set model to evaluation mode
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    frame_interval = total_frames // num_frames  # Calculate the interval to sample frames evenly
    
    efficientface_features = []
    
    for i in range(num_frames):
        # Set the video frame position based on the interval
        frame_position = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to extract frame {frame_position} from {video_path}")
            continue
        
        # Preprocess the frame for EfficientNet
        input_frame = cv2.resize(frame, (224, 224))  # Resize the frame
        input_frame = input_frame.transpose(2, 0, 1)  # Convert to (C, H, W)
        input_frame = torch.tensor(input_frame).unsqueeze(0).float() / 255.0  # Normalize
        
        # Extract features using EfficientNet
        with torch.no_grad():
            features = model(input_frame).numpy()  # Extract features
            efficientface_features.append(features)
    
    cap.release()
    return np.array(efficientface_features).squeeze()

def preprocess_all_videos(source_dir, processed_output, target_dir):
    if not os.path.exists(processed_output):
        os.makedirs(processed_output)

    audio_features, visual_features, labels_list = [], [], []
     # Get the maximum number of frames across all videos
    max_frames = max_frames_in_directory(source_dir)

    # Iterate over all files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for filename in tqdm(files, desc="Processing videos"):
            if filename.endswith(".avi"):
                video_path = os.path.join(root, filename)
                audio_path = video_path.replace(".avi", ".wav")

                # Extract the emotion label from the filename
                emotion_code = filename.split('_')[1][0]
                label = label_map.get(emotion_code)
                if label is None:
                    print(f"Unknown label for file: {filename}")
                    continue

                # Extract visual features
                openface_feat = extract_visual_features_openface(video_path, processed_output)
                efficientface_features = extract_efficientface_features(video_path, openface_feat.shape[0])
                visual_feat = np.concatenate((openface_feat, efficientface_features.squeeze()), axis=1)
    
                if visual_feat is None:
                    continue

                # Extract audio features
                audio_feat = extract_audio_features(audio_path)

                visual_feat = pad_symmetric(visual_feat, max_frames)
                audio_feat = pad_audio_symmetric(audio_feat, max_frames)

                # Store the features and label
                audio_features.append(audio_feat)
                visual_features.append(visual_feat)
                labels_list.append(label)

    # Save the extracted features and labels
    np.save(os.path.join(target_dir, 'audio_features.npy'), np.array(audio_features))
    np.save(os.path.join(target_dir, 'visual_features.npy'), np.array(visual_features))
    np.save(os.path.join(target_dir, 'labels.npy'), np.array(labels_list))

    print(f"Extracted features and labels have been saved to {target_dir}")

# Usage
source_dir = 'all_videos'  # Folder containing the .avi files
processed_output = './Processed/'  # Folder to store temporary OpenFace outputs
target_dir = './ExtractedFeatures/'  # Folder to save the final features and labels

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

preprocess_all_videos(source_dir, processed_output, target_dir)
