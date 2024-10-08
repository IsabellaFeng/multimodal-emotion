import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def k_fold_split_data(visual_features, audio_features, labels, k=5):
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=2024)
    
    fold = 1
    
    # Loop through each fold
    for train_index, val_index in kf.split(labels):
        print(f"Processing fold {fold}...")
        
        # Training data for the current fold
        train_visual = visual_features[train_index]
        train_audio = audio_features[train_index]
        train_labels = labels[train_index]
        
        # Validation data for the current fold
        val_visual = visual_features[val_index]
        val_audio = audio_features[val_index]
        val_labels = labels[val_index]
        
        # Save each fold's data
        output_dir = f'ExtractedFeatures/fold_{fold}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        np.save(os.path.join(output_dir, 'train_audio.npy'), train_audio)
        np.save(os.path.join(output_dir, 'train_visual.npy'), train_visual)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)

        np.save(os.path.join(output_dir, 'val_audio.npy'), val_audio)
        np.save(os.path.join(output_dir, 'val_visual.npy'), val_visual)
        np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)

        print(f"Data for fold {fold} saved to directory '{output_dir}'")
        
        fold += 1

def shuffle_and_split_data(visual_features, audio_features, labels, test_size=1/6):
    np.random.seed(2024)
    
    # Split data into training + validation (5/6) and test sets (1/6)
    train_visual, test_visual, train_audio, test_audio, train_labels, test_labels = train_test_split(
        visual_features, audio_features, labels, test_size=test_size, random_state=2024
    )
    
    return (train_audio, train_visual, train_labels), (test_audio, test_visual, test_labels)

if __name__ == "__main__":
    # Load the feature files
    visual_features = np.load('ExtractedFeatures/visual_features.npy', allow_pickle=True)
    audio_features = np.load('ExtractedFeatures/audio_features.npy', allow_pickle=True)
    labels = np.load('ExtractedFeatures/labels.npy', allow_pickle=True)

    # First, split the data into train + validation (5/6) and test sets (1/6)
    (train_audio, train_visual, train_labels), \
    (test_audio, test_visual, test_labels) = shuffle_and_split_data(visual_features, audio_features, labels, test_size=1/6)

    # Save the test set separately
    output_dir = 'ExtractedFeatures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'test_audio.npy'), test_audio)
    np.save(os.path.join(output_dir, 'test_visual.npy'), test_visual)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

    print(f"Test set saved to directory '{output_dir}'")

    # Perform K-Fold Cross-Validation on the training + validation data (5/6 of the data)
    k_fold_split_data(train_visual, train_audio, train_labels, k=5)

    print("K-Fold Cross-Validation data split and saved.")
