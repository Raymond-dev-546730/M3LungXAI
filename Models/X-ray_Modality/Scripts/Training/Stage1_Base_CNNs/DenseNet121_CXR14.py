# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import os
import pandas as pd
import random

# RANDOM SEED: 442

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define Focal Loss to handle class imbalance
class focal_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def train_model():

    set_seed(442)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set paths
    data_dir = "./CXR14_processed_224x224"

    # Data augmentation and transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    # Calculate class weights to handle imbalance
    class_counts = [
        len([entry for entry in os.scandir(os.path.join(data_dir, class_name)) if entry.is_file()]) 
        for class_name in ['nodule', 'normal']
    ]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float).to(device)

    # Splitting the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create a mapping from indices to class labels for the train dataset
    train_labels = [full_dataset.targets[idx] for idx in train_dataset.indices]
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Weighted random sampler for imbalanced classes
    train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    # Data loaders
    train_loader = DataLoader(
        Subset(full_dataset, train_dataset.indices), 
        batch_size=32, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_dataset.indices),
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model = model.to(device)

    # Use Focal Loss
    criterion = focal_loss(alpha=1.0, gamma=2.0).to(device)

    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Cyclic LR Scheduler
    scheduler = CyclicLR(
        optimizer, 
        base_lr=1e-6, 
        max_lr=1e-4, 
        step_size_up=5, 
        mode='triangular2'
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda')

    # Early stopping parameters
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_auc = float('-inf')
    min_delta = 1e-4

    # Directory to save models
    model_save_dir = "DenseNet121_CXR14"
    os.makedirs(model_save_dir, exist_ok=True)

    # List to store metrics for CSV
    metrics = []

    num_epochs = 75

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_targets = []
        train_outputs = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            softmax_outputs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            
            train_outputs.extend(softmax_outputs)
            train_targets.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_outputs = np.array(train_outputs)
        train_targets = np.array(train_targets)
        
        # Training metrics
        train_auc = roc_auc_score(train_targets, train_outputs[:, 1])
        train_preds = np.argmax(train_outputs, axis=1)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        train_precision = precision_score(train_targets, train_preds, average='macro', zero_division=1)
        train_recall = recall_score(train_targets, train_preds, average='macro')
        train_accuracy = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_outputs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                softmax_outputs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                val_outputs.extend(softmax_outputs)
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_outputs = np.array(val_outputs)
        val_targets = np.array(val_targets)

        # Validation metrics
        val_auc = roc_auc_score(val_targets, val_outputs[:, 1])
        val_preds = np.argmax(val_outputs, axis=1)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_precision = precision_score(val_targets, val_preds, average='macro', zero_division=1)
        val_recall = recall_score(val_targets, val_preds, average='macro')
        val_accuracy = accuracy_score(val_targets, val_preds)

        # Print logs
        print(f"Epoch {epoch}/{num_epochs}")
        print('-' * 10)
        print(
            f"train Loss: {train_loss:.4f} "
            f"Acc: {train_accuracy:.4f} "
            f"AUC: {train_auc:.4f} "
            f"Recall: {train_recall:.4f} "
            f"Precision: {train_precision:.4f} "
            f"F1: {train_f1:.4f}"
        )
        print(
            f"val   Loss: {val_loss:.4f} "
            f"Acc: {val_accuracy:.4f} "
            f"AUC: {val_auc:.4f} "
            f"Recall: {val_recall:.4f} "
            f"Precision: {val_precision:.4f} "
            f"F1: {val_f1:.4f}"
        )

        # Save metrics to list for CSV
        metrics.append({
            'Phase': 'train',
            'Loss': train_loss,
            'Accuracy': train_accuracy,
            'AUC': train_auc,
            'Recall': train_recall,
            'Precision': train_precision,
            'F1': train_f1
        })
        metrics.append({
            'Phase': 'val',
            'Loss': val_loss,
            'Accuracy': val_accuracy,
            'AUC': val_auc,
            'Recall': val_recall,
            'Precision': val_precision,
            'F1': val_f1
        })

        # Early stopping check
        if (val_auc - best_val_auc) > min_delta:
            best_val_auc = val_auc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Step the scheduler
        scheduler.step()

        # Save model at each epoch
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'epoch_{epoch}_model.pth'))

    metrics_df = pd.DataFrame(metrics, columns=['Phase','Loss','Accuracy','AUC','Recall','Precision','F1'])
    metrics_df.to_csv(os.path.join(model_save_dir, 'DenseNet121_Metrics_CXR14.csv'), index=False)

    print("Training complete.")

if __name__ == "__main__":
    train_model()
