import numpy as np
import os

def split_data (visual_features, audio_features, labels, actors):
    unique_actors = np.unique(actors)
    train_actors = unique_actors[:61]
    val_actors = unique_actors[61:76]
    test_actors = unique_actors[76:]

    # Create masks for selecting data corresponding to these actors
    train_mask = np.isin(actors, train_actors)
    val_mask = np.isin(actors, val_actors)
    test_mask = np.isin(actors, test_actors)

    # Split the data based on the masks
    train_audio = audio_features[train_mask]
    train_visual = visual_features[train_mask]
    train_labels = labels[train_mask]

    val_audio = audio_features[val_mask]
    val_visual = visual_features[val_mask]
    val_labels = labels[val_mask]

    test_audio = audio_features[test_mask]
    test_visual = visual_features[test_mask]
    test_labels = labels[test_mask]

    return (train_audio, train_visual, train_labels), \
        (val_audio, val_visual, val_labels), \
        (test_audio, test_visual, test_labels)



if __name__ == "__main__":
    # Load the feature files
    visual_features = np.load('ExtractedFeatures/visual_features.npy', allow_pickle=True)
    audio_features = np.load('ExtractedFeatures/audio_features.npy', allow_pickle=True)
    labels = np.load('ExtractedFeatures/labels.npy', allow_pickle=True)
    actors = np.load('ExtractedFeatures/actor.npy', allow_pickle=True)


    # Shuffle and split the data
    (train_audio, train_visual, train_labels), \
    (val_audio, val_visual, val_labels), \
    (test_audio, test_visual, test_labels) = split_data(visual_features, audio_features, labels, actors)

    # Create a new directory for the split datasets
    output_dir = 'ExtractedFeatures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the split datasets in the new directory
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
