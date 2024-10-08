import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from train import ClassificationModel  
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import random
import joblib

def load_test_data(apply_pca=True, pca_components=100):
    # Load your test data
    test_visual = np.load('ExtractedFeatures/test_visual.npy', allow_pickle=True)
    test_audio = np.load('ExtractedFeatures/test_audio.npy', allow_pickle=True)
    test_audio = test_audio.transpose(0, 2, 1)  # Swap feature and frames dimension
    test_labels = np.load('ExtractedFeatures/test_labels.npy', allow_pickle=True)

    # Standardize visual data
    mean_visual = np.mean(test_visual)
    std_visual = np.std(test_visual)
    if std_visual == 0:
        std_visual = 1  # Avoid division by zero
    test_visual = (test_visual - mean_visual) / std_visual

    # Standardize audio data
    mean_audio = np.mean(test_audio)
    std_audio = np.std(test_audio)
    if std_audio == 0:
        std_audio = 1  # Avoid division by zero
    test_audio = (test_audio - mean_audio) / std_audio

    if apply_pca:

        pca_audio = joblib.load('pca_audio_model.pkl')  # Load the PCA models
        pca_visual = joblib.load('pca_visual_model.pkl')

        # Flatten the test audio data to apply PCA
        test_audio_flattened = test_audio.reshape(test_audio.shape[0] * test_audio.shape[1], -1)
        test_audio_reduced_flat = pca_audio.transform(test_audio_flattened)   
        # Reshape the reduced audio back to match the original sequence length
        test_audio_reduced = test_audio_reduced_flat.reshape(test_audio.shape[0], test_audio.shape[1], -1)

        # Apply PCA to visual features
        test_visual_flattened = test_visual.reshape(test_visual.shape[0] * test_visual.shape[1], -1)
        test_visual_reduced_flat = pca_visual.transform(test_visual_flattened)
        # Reshape the reduced visual back to match the original sequence length
        test_visual_reduced = test_visual_reduced_flat.reshape(test_visual.shape[0], test_visual.shape[1], -1)
    else:
        test_audio_reduced = test_audio
        test_visual_reduced = test_visual

    # Concatenate visual and audio features along the feature dimension (axis=2)
    test_combined = np.concatenate((test_visual_reduced, test_audio_reduced), axis=2)

    # Convert to PyTorch tensors
    test_tensor = torch.tensor(test_combined, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = TensorDataset(test_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader

def evaluate_model(test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = ClassificationModel(
        input_size=400,
        hidden_size=256,
        num_heads=4,
        num_layers=8,
        num_classes=7,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load('best_model_parameters.pth'))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)  # Get model outputs
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

if __name__ == "__main__":
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_loader = load_test_data(apply_pca=True, pca_components=200)
    evaluate_model(test_loader)