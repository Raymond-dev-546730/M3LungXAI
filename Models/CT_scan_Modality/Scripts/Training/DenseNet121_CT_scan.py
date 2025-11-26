# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import time
import copy
import os
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import random

# Set random seeds for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_random_seeds(42)

# Define data directories
data_dir = './CT_scan_processed_128x128' 

# Define model name
model_name = 'DenseNet121'

# Define transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load datasets
image_dataset = datasets.ImageFolder(data_dir)

# Split the data into train, val, and test sets
train_idx, test_idx = train_test_split(
    np.arange(len(image_dataset.targets)),
    test_size=0.1,  
    stratify=image_dataset.targets,
    random_state=42
)

# Further split the training set for validation
train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.1 / 0.9,  # 0.1 / 0.9 of the remaining data
    stratify=[image_dataset.targets[i] for i in train_idx],
    random_state=42
)

# Create subsets with transforms applied
train_subset = Subset(image_dataset, train_idx)
val_subset = Subset(image_dataset, val_idx)
test_subset = Subset(image_dataset, test_idx)

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['test'])

train_dataset.samples = [train_dataset.samples[i] for i in train_idx]
val_dataset.samples = [val_dataset.samples[i] for i in val_idx]
test_dataset.samples = [test_dataset.samples[i] for i in test_idx]

# Define class weights to handle class imbalance for training subset
class_counts = np.bincount([image_dataset.targets[i] for i in train_idx])
class_weights = 1. / class_counts
samples_weights = [class_weights[image_dataset.targets[i]] for i in train_idx]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

# Create dataloaders
batch_size = 64
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=16),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

# Initialize the model
def initialize_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5), # Drop out layer of 0.5
        nn.Linear(num_ftrs, 4)
    )
    return model

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function with class weights
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Create directory to save epoch weights
weights_dir = f'./{model_name}_CT_Scan'
os.makedirs(weights_dir, exist_ok=True)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, fold=0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    metrics = {
        'train': {'accuracy': [], 'auc': [], 'recall': [], 'precision': [], 'f1': [], 'loss': []},
        'val': {'accuracy': [], 'auc': [], 'recall': [], 'precision': [], 'f1': [], 'loss': []}
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []
            all_probs = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                probs = nn.functional.softmax(outputs, dim=1)  # Ensure softmax is applied

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')

            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['accuracy'].append(epoch_acc.item() if torch.is_tensor(epoch_acc) else epoch_acc)
            metrics[phase]['auc'].append(epoch_auc)
            metrics[phase]['recall'].append(epoch_recall)
            metrics[phase]['precision'].append(epoch_precision)
            metrics[phase]['f1'].append(epoch_f1)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f} Recall: {epoch_recall:.4f} Precision: {epoch_precision:.4f} F1: {epoch_f1:.4f}')

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(weights_dir, f'{model_name}_fold_{fold}_best.pth'))  # Save only the best model for each fold

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F1: {best_f1:.4f}')

    model.load_state_dict(best_model_wts)
    return model, metrics, best_f1

def cross_validate_model(dataset, num_folds=5, num_epochs=25):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    all_metrics = {
        'train': {'accuracy': [], 'auc': [], 'recall': [], 'precision': [], 'f1': [], 'loss': []},
        'val': {'accuracy': [], 'auc': [], 'recall': [], 'precision': [], 'f1': [], 'loss': []}
    }

    best_overall_f1 = 0.0
    best_fold = 0

    targets = [dataset.dataset.targets[i] for i in dataset.indices]

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f'Fold {fold + 1}/{num_folds}')

        train_subset = Subset(dataset.dataset, [dataset.indices[i] for i in train_idx])
        val_subset = Subset(dataset.dataset, [dataset.indices[i] for i in val_idx])

        train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val'])

        train_dataset.samples = [train_dataset.samples[i] for i in train_idx]
        val_dataset.samples = [val_dataset.samples[i] for i in val_idx]

        class_counts = np.bincount([image_dataset.targets[i] for i in train_idx])
        class_weights = 1. / class_counts
        samples_weights = [class_weights[image_dataset.targets[i]] for i in train_idx]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        model = initialize_model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)

        model, metrics, best_f1 = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs, fold)

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_fold = fold

        for phase in ['train', 'val']:
            for metric in metrics[phase]:
                all_metrics[phase][metric].extend(metrics[phase][metric])

    # After all folds, load the best model weights from the fold with the highest F1 score
    best_model_wts_path = os.path.join(weights_dir, f'{model_name}_fold_{best_fold}_best.pth')
    
    return all_metrics, best_model_wts_path

if __name__ == '__main__':
    print("Starting cross-validation...")
    all_metrics, best_model_wts_path = cross_validate_model(train_subset, num_folds=5, num_epochs=25)

    metrics_df = pd.DataFrame({
        'Phase': ['train'] * len(all_metrics['train']['loss']) + ['val'] * len(all_metrics['val']['loss']),
        'Loss': all_metrics['train']['loss'] + all_metrics['val']['loss'],
        'Accuracy': all_metrics['train']['accuracy'] + all_metrics['val']['accuracy'],
        'AUC': all_metrics['train']['auc'] + all_metrics['val']['auc'],
        'Recall': all_metrics['train']['recall'] + all_metrics['val']['recall'],
        'Precision': all_metrics['train']['precision'] + all_metrics['val']['precision'],
        'F1': all_metrics['train']['f1'] + all_metrics['val']['f1']
    })

    metrics_df.to_csv('./DenseNet121_CT_Scan/densenet121_cv_metrics_CT.csv', index=False) 
    print("Cross-validation metrics saved to 'densenet121_cv_metrics_CT.csv'.")

    print("Evaluating on the test set...")

    final_model = initialize_model().to(device)  
    state_dict = torch.load(best_model_wts_path, weights_only=True) 
    final_model.load_state_dict(state_dict)  
    final_model.eval()  

    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = final_model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            probs = nn.functional.softmax(outputs, dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    test_loss = running_loss / dataset_sizes['test']
    test_acc = running_corrects.double() / dataset_sizes['test']
    test_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    test_metrics = {
        'loss': test_loss,
        'accuracy': test_acc.cpu().item(),
        'auc': test_auc,
        'recall': test_recall,
        'precision': test_precision,
        'f1': test_f1
    }

    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv('./DenseNet121_CT_Scan/densenet121_test_metrics_CT.csv', index=False)
    print("Test metrics saved to 'densenet121_test_metrics_CT.csv'.")

    # Class 0 = Adenocarcinoma
    # Class 1 = Large_Cell_Carcinoma
    # Class 2 = Normal
    # Class 3 = Squamous_Cell_Carcinoma

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f} Recall: {test_recall:.4f} Precision: {test_precision:.4f} F1: {test_f1:.4f}')
