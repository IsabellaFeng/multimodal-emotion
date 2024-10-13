## Requirements

The project requires the following Python packages:
Python 3.9.5
numpy
opencv-python
librosa
pandas
tqdm
transformers
torch
efficientnet-pytorch
torchvision
scikit-learn
joblib
tensorboard

## Project Structure
```
├── CREMA-D/                      # Folder containing model and scripts for the CREMA-D dataset
│   ├── ExtractedFeatures/        # Folder containing extracted features for the dataset
│   └── scripts/                  # Folder containing scripts to process and train models
│       ├── convert-avi.py        # Script to convert files to .avi format
│       ├── preprocess.py         # Script to preprocess data for the model
│       ├── split_data.py         # Script to split the dataset into training and validation
│       ├── test.py               # Script to test the model on the dataset
│       ├── train.py              # Script to train the model on the dataset
│   ├── best_model_parameters.pth # Saved model parameters
│   ├── pca_audio_model.pkl       # Saved PCA model for audio features
│   ├── pca_visual_model.pkl      # Saved PCA model for visual features
│
├── RAVDESS/                      # Folder containing model and scripts for the RAVDESS dataset
│   ├── ExtractedFeatures/        # Folder containing extracted features for the dataset
│   └── scripts/                  # Folder containing scripts to process and train models
│       ├── convert-avi.py        # Script to convert files to .avi format
│       ├── preprocess.py         # Script to preprocess data for the model
│       ├── split_data.py         # Script to split the dataset into training and validation
│       ├── test.py               # Script to test the model on the dataset
│       ├── train.py              # Script to train the model on the dataset
│   ├── best_model_parameters.pth # Saved model parameters
│   ├── pca_audio_model.pkl       # Saved PCA model for audio features
│   ├── pca_visual_model.pkl      # Saved PCA model for visual features
│
├── SAVEE/                        # Folder containing model and scripts for the SAVEE dataset
│   ├── ExtractedFeatures/        # Folder containing extracted features for the dataset
│   └── scripts/                  # Folder containing scripts to process and train models
│       ├── convert-avi.py        # Script to convert files to .avi format
│       ├── preprocess.py         # Script to preprocess data for the model
│       ├── split_data.py         # Script to split the dataset into training and validation
│       ├── test.py               # Script to test the model on the dataset
│       ├── train.py              # Script to train the model on the dataset
│   ├── best_model_parameters.pth # Saved model parameters
│   ├── pca_audio_model.pkl       # Saved PCA model for audio features
│   ├── pca_visual_model.pkl      # Saved PCA model for visual features
└── README.md                     # This file

```


## train and test
for each of the dataset, you can navigate to the folder and python scripts/train.py to train the data and generate the best_model_parameters.pth and then run python scripts/test.py to test the accuracy based on the test set. It will use the extracted featured in ExtractedFeatures folder

## train by yourself
if you want to try extract the features by yourself, you will need to split data in the correct structure:
- RAVDESS:
    Create a folder in root directory and then split 24 speech authors in folders named ACTOR01 - ACTOR24, each foler should contain this actor's audio . mp4 file and vedip .wav file. you can update the path in script/preprocess.py

- SAVEE:
    Save all the videos in /all_videos or modify in the, you can update the path in script/preprocess.py.

- CREMA-D:
    Save all the videos in /avi or create a new path and update preprocess.py

    
