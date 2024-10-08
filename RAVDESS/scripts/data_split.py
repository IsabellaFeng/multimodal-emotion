import os
import numpy as np
import cv2

def split_data(visual_features, audio_features, labels, actors_list):
    # Define actors for each set
    train_actors = [f'ACTOR{i:02d}' for i in range(9, 25)]
    val_actors = ['ACTOR05', 'ACTOR06', 'ACTOR07', 'ACTOR08']
    test_actors = ['ACTOR01', 'ACTOR02', 'ACTOR03', 'ACTOR04']

    # Split data based on actor
    train_idx = [i for i, actor in enumerate(actors_list) if actor in train_actors]
    val_idx = [i for i, actor in enumerate(actors_list) if actor in val_actors]
    test_idx = [i for i, actor in enumerate(actors_list) if actor in test_actors]

    train_audio = audio_features[train_idx]
    train_visual = visual_features[train_idx]
    train_labels = labels[train_idx]

    val_audio = audio_features[val_idx]
    val_visual = visual_features[val_idx]
    val_labels = labels[val_idx]

    test_audio = audio_features[test_idx]
    test_visual = visual_features[test_idx]
    test_labels = labels[test_idx]

    return (train_audio, train_visual, train_labels), (val_audio, val_visual, val_labels), (test_audio, test_visual, test_labels)

if __name__ == "__main__":
    # Load the new feature files
    visual_features = np.load('ExtractedFeatures/visual_features.npy', allow_pickle=True)
    audio_features = np.load('ExtractedFeatures/audio_features.npy', allow_pickle=True)
    labels = np.load('ExtractedFeatures/labels.npy', allow_pickle=True)
    actors = np.load('ExtractedFeatures/actors.npy', allow_pickle=True)

    # Split the data
    (train_audio, train_visual, train_labels), \
    (val_audio, val_visual, val_labels), \
    (test_audio, test_visual, test_labels) = split_data(visual_features, audio_features, labels, actors)

    # Create a new directory for the new files
    output_dir = 'ExtractedFeatures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the split datasets with the new names in the new directory
    np.save(os.path.join(output_dir, 'train_audio.npy'), train_audio)
    np.save(os.path.join(output_dir, 'train_visual.npy'), train_visual)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)

    np.save(os.path.join(output_dir, 'val_audio.npy'), val_audio)
    np.save(os.path.join(output_dir, 'val_visual.npy'), val_visual)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)

    np.save(os.path.join(output_dir, 'test_audio.npy'), test_audio)
    np.save(os.path.join(output_dir, 'test_visual.npy'), test_visual)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

    print(f"Data split and saved to directory '{output_dir}'")
