import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import random
import math
import os
from statistics import mean, stdev
import joblib

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in) 
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_size, hidden_size // num_heads, hidden_size // num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size * 4, dropout)

    def forward(self, x, mask=None):
        x, attn = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x, attn

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0.1):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.input_linear(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, feature)
        attn_list = []
        for layer in self.transformer_layers:
            x, attn = layer(x)
            attn_list.append(attn)
        x = x.mean(dim=0)  # Reduce dimensions by averaging over the sequence
        return x, attn_list

class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, num_classes, dropout=0.1):
        super(ClassificationModel, self).__init__()
        self.transformer_feature_extractor = TransformerFeatureExtractor(input_size, hidden_size, num_heads, num_layers, dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        features, attn_list = self.transformer_feature_extractor(x)
        output = self.fc(features)
        return output, attn_list

from sklearn.decomposition import PCA

def process_data(train_visual, train_audio, val_visual, val_audio, train_labels, val_labels, apply_pca=True, pca_components=200 ):
    train_visual = train_visual.reshape(train_visual.shape[0], -1, train_visual.shape[-1])  # Flatten spatial dimensions
    val_visual = val_visual.reshape(val_visual.shape[0], -1, val_visual.shape[-1])  # Flatten spatial dimensions

    train_audio = train_audio.transpose(0, 2, 1)  # Swap feature and frames dimension
    val_audio = val_audio.transpose(0, 2, 1)  # Swap feature and frames dimension

    # Standardize visual data
    mean_visual = np.mean(train_visual)
    std_visual = np.std(train_visual)
    train_visual = (train_visual - mean_visual) / std_visual
    val_visual = (val_visual - mean_visual) / std_visual
    
    # Standardize audio data
    mean_audio = np.mean(train_audio)
    std_audio = np.std(train_audio)
    train_audio = (train_audio - mean_audio) / std_audio
    val_audio = (val_audio - mean_audio) / std_audio

    if apply_pca:
        # Flatten the train and val audio data to apply PCA
        train_audio_flattened = train_audio.reshape(train_audio.shape[0] * train_audio.shape[1], -1)
        val_audio_flattened = val_audio.reshape(val_audio.shape[0] * val_audio.shape[1], -1)
        
        # Apply PCA to audio features
        pca_audio = PCA(n_components=pca_components)
        train_audio_reduced_flat = pca_audio.fit_transform(train_audio_flattened)
        val_audio_reduced_flat = pca_audio.transform(val_audio_flattened)
        
        # Reshape the reduced audio back to match the original sequence length
        train_audio_reduced = train_audio_reduced_flat.reshape(train_audio.shape[0], train_audio.shape[1], -1)
        val_audio_reduced = val_audio_reduced_flat.reshape(val_audio.shape[0], val_audio.shape[1], -1)

        train_visual_flattened = train_visual.reshape(train_visual.shape[0] * train_visual.shape[1], -1)
        val_visual_flattened = val_visual.reshape(val_visual.shape[0] * val_visual.shape[1], -1)

        pca_visual = PCA(n_components=pca_components)
        train_visual_reduced_flat = pca_visual.fit_transform(train_visual_flattened)
        val_visual_reduced_flat = pca_visual.transform(val_visual_flattened)

        # Reshape the reduced visual back to match the original sequence length
        train_visual_reduced = train_visual_reduced_flat.reshape(train_visual.shape[0], train_visual.shape[1], -1)
        val_visual_reduced = val_visual_reduced_flat.reshape(val_visual.shape[0], val_visual.shape[1], -1)
        joblib.dump(pca_audio, 'pca_audio_model.pkl')
        joblib.dump(pca_visual, 'pca_visual_model.pkl')
    else:
        train_audio_reduced = train_audio
        val_audio_reduced = val_audio
        train_visual_reduced = train_visual
        val_visual_reduced = val_visual


    # Concatenate visual and (possibly reduced) audio features
    train_combined = np.concatenate((train_visual_reduced, train_audio_reduced), axis=2)
    val_combined = np.concatenate((val_visual_reduced, val_audio_reduced), axis=2)

    train_combined_tensor = torch.tensor(train_combined, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_combined_tensor = torch.tensor(val_combined, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_combined_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(val_combined_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"{train_combined.shape}")
    return train_loader, val_loader

def load_data_for_fold(fold_num):
    """
    Load training and validation data for the specified fold.
    """
    fold_dir = f'ExtractedFeatures/fold_{fold_num}'

    # Load the data
    train_visual = np.load(os.path.join(fold_dir, 'train_visual.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(fold_dir, 'train_labels.npy'), allow_pickle=True)
    val_visual = np.load(os.path.join(fold_dir, 'val_visual.npy'), allow_pickle=True)
    val_labels = np.load(os.path.join(fold_dir, 'val_labels.npy'), allow_pickle=True)

    train_audio = np.load(os.path.join(fold_dir, 'train_audio.npy'), allow_pickle=True)
    val_audio = np.load(os.path.join(fold_dir, 'val_audio.npy'), allow_pickle=True)

    # Convert to PyTorch tensors
    train_loader, val_loader = process_data(train_visual, train_audio, val_visual,val_audio, train_labels, val_labels)
    return train_loader, val_loader

def train_model(train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationModel(
        input_size=400,  # combined feature size (flattened visual + flattened audio)
        hidden_size=256,
        num_heads=4,
        num_layers=8,
        num_classes=7,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    writer = SummaryWriter()
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    best_val_loss = float('inf')
    best_val_accuracy = 0
    num_epochs = 30
    grad_clip_value = 1.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_samples = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_samples += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_accuracy = 100 * correct_samples / total_samples
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_accuracy'].append(train_accuracy)
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = 100 * val_correct / val_total
        average_val_loss = val_loss / len(val_loader)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        history['val_loss'].append(average_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1_score'].append(f1)

        writer.add_scalar('Loss/validation', average_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('Precision/validation', precision, epoch)
        writer.add_scalar('Recall/validation', recall, epoch)
        writer.add_scalar('F1_Score/validation', f1, epoch)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model_parameters.pth')

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    writer.close()
    print(f"Best Validation Loss: {best_val_loss:.4f}, Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print("Training complete")

    return history  # Return the history dictionary to be used for aggregation

def k_fold_train(k_folds):
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'val_loss': []
    }

    for fold_num in range(1, k_folds + 1):
        print(f"Starting training for fold {fold_num}...")
        train_loader, val_loader = load_data_for_fold(fold_num)
        fold_history = train_model(train_loader, val_loader)

        # Collect results from this fold
        fold_results['accuracy'].append(fold_history['val_accuracy'][-1])
        fold_results['precision'].append(fold_history['precision'][-1])
        fold_results['recall'].append(fold_history['recall'][-1])
        fold_results['f1_score'].append(fold_history['f1_score'][-1])
        fold_results['val_loss'].append(fold_history['val_loss'][-1])

    # Calculate mean and standard deviation for each metric
    aggregated_results = { 
        metric: {
            'mean': mean(values),
            'std': stdev(values)
        } for metric, values in fold_results.items()
    }

    print("\nFinal K-Fold Cross-Validation Results:")
    for metric, values in aggregated_results.items():
        print(f"{metric.capitalize()} - Mean: {values['mean']:.4f}, Std: {values['std']:.4f}")

if __name__ == "__main__":
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    k_folds = 5
    k_fold_train(k_folds)