# Import required libraries
import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

# Prevent issues related to parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Random Seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Dataset preparation 
class spanned_dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_distribution = []  

        # Read the JSONL index file
        with open(jsonl_path, "r") as index_file:
            for line in index_file:
                entry = json.loads(line.strip())
                annotations_path = entry["annotations_path"]

                with open(annotations_path, "r") as labeled_file:
                    labeled_data = json.load(labeled_file)
                    text = labeled_data["text"]
                    spans = labeled_data["spans"]

                    tokenized = tokenizer(
                        text,
                        return_offsets_mapping=True,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                    )
                    offset_mapping = tokenized.pop("offset_mapping")
                    labels = [0] * len(tokenized["input_ids"])

                    for span in spans:
                        if span["label"] == "SYMPTOM":
                            start, end = span["start"], span["end"]
                            token_start = next((i for i, offset in enumerate(offset_mapping) if offset[0] <= start < offset[1]), None)
                            token_end = next((i for i, offset in enumerate(offset_mapping) if offset[0] < end <= offset[1]), None)

                            if token_start is not None and token_end is not None:
                                for i in range(token_start, token_end + 1):
                                    labels[i] = 1

                    self.examples.append({
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                        "labels": labels[: self.max_length],
                    })

                    self.labels_distribution.append(sum(labels) / len(labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

    def get_stratification_labels(self):
        return np.digitize(self.labels_distribution, bins=np.linspace(0, 1, 10))

# Model initialization function 
def init_model():
    model = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-large-cased", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

# Training/validation epoch
def epoch_cycle(model, loader, optimizer=None, scheduler=None, train=False, accumulation_steps=4):
    mode = "Train" if train else "Validation"
    print(f"\n{mode}")
    model.train() if train else model.eval()
    device = next(model.parameters()).device
    scaler = GradScaler()
    total_loss = 0
    all_predictions, all_labels = [], []
    all_probs = []  # For ROC-AUC
    running_loss = 0

    if train:
        optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].long().to(device)

        with autocast(device_type="cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            if train:
                loss = loss / accumulation_steps

        if train:
            scaler.scale(loss).backward()
            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        running_loss += loss.item() * (accumulation_steps if train else 1)
        
        # Get predictions and probabilities for ROC-AUC
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1).view(-1).cpu().numpy()
            true_labels = labels.view(-1).cpu().numpy()
            prob_positive = probs[:, :, 1].view(-1).cpu().numpy()  # Probability of positive class
            
            # Filter out padding tokens 
            mask = attention_mask.view(-1).cpu().numpy() == 1
            all_predictions.extend(predictions[mask])
            all_labels.extend(true_labels[mask])
            all_probs.extend(prob_positive[mask])

        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Batch {batch_idx + 1}/{len(loader)} - Average Loss: {avg_loss:.4f}")

    metrics = {
        'precision': precision_score(all_labels, all_predictions, average="weighted", zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average="weighted", zero_division=0),
        'f1': f1_score(all_labels, all_predictions, average="weighted", zero_division=0),
        'accuracy': accuracy_score(all_labels, all_predictions),
        'loss': running_loss / len(loader)
    }
    
    # Calculate ROC-AUC
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs, average="weighted")
    else:
        metrics['roc_auc'] = 0.0
        print(f"Warning: Only one class present in {mode} labels, ROC-AUC set to 0.0")

    print(f"{mode} Metrics:\nPrecision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics

# Load Dataset
jsonl_path = "dataset_index.jsonl"
tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
dataset = spanned_dataset(jsonl_path, tokenizer)

# Separate test set
indices = list(range(len(dataset)))
stratify_labels = dataset.get_stratification_labels()
train_val_indices, test_indices = train_test_split(
    indices, test_size=0.15, random_state=42,
    stratify=stratify_labels
)

# Create test dataset
test_dataset = torch.utils.data.Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize cross-validation
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_val_stratify_labels = stratify_labels[train_val_indices]

# Store results for each fold
fold_results = []
best_val_f1 = 0
best_val_roc_auc = 0
best_model = None
best_fold = None

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, train_val_stratify_labels)):
    print(f"Fold {fold + 1}/{n_splits}")
    print(f"{'-'*50}")
    
    # Get actual indices from train_val_indices
    train_indices = [train_val_indices[i] for i in train_idx]
    val_indices = [train_val_indices[i] for i in val_idx]
    
    # Create datasets for this fold
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model and optimization components
    model = init_model()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    accumulation_steps = 4
    epochs = 3
    
    num_training_steps = (len(train_loader) // accumulation_steps) * epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    fold_best_val_f1 = 0
    fold_best_val_roc_auc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 10)
        
        train_metrics = epoch_cycle(
            model, train_loader,
            optimizer=optimizer,
            train=True,
            accumulation_steps=accumulation_steps
        )
        lr_scheduler.step()
        
        with torch.no_grad():
            val_metrics = epoch_cycle(model, val_loader)
            
        if val_metrics['f1'] > fold_best_val_f1:
            fold_best_val_f1 = val_metrics['f1']
            fold_best_val_roc_auc = val_metrics['roc_auc']
    
    # Save best model based on validation performance
    if fold_best_val_f1 > best_val_f1:
        best_val_f1 = fold_best_val_f1
        best_val_roc_auc = fold_best_val_roc_auc
        best_model = model
        best_fold = fold + 1
    
    # Evaluate on test set 
    print(f"\nEvaluating Fold {fold + 1} on Test Set")
    with torch.no_grad():
        test_metrics = epoch_cycle(model, test_loader)
    
    fold_results.append(test_metrics)

# Calculate and print aggregate results
avg_metrics = {
    'precision': np.mean([result['precision'] for result in fold_results]),
    'recall': np.mean([result['recall'] for result in fold_results]),
    'f1': np.mean([result['f1'] for result in fold_results]),
    'accuracy': np.mean([result['accuracy'] for result in fold_results]),
    'roc_auc': np.mean([result['roc_auc'] for result in fold_results])
}

std_metrics = {
    'precision': np.std([result['precision'] for result in fold_results]),
    'recall': np.std([result['recall'] for result in fold_results]),
    'f1': np.std([result['f1'] for result in fold_results]),
    'accuracy': np.std([result['accuracy'] for result in fold_results]),
    'roc_auc': np.std([result['roc_auc'] for result in fold_results])
}

print(f"\nAverage Metrics for Test Set")
print(f"Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
print(f"Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
print(f"F1-Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
print(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f} ± {std_metrics['roc_auc']:.4f}")

print(f"\nBest model selected from Fold {best_fold} based on validation performance:")
print(f"Validation F1-Score: {best_val_f1:.4f}")
print(f"Validation ROC-AUC: {best_val_roc_auc:.4f}")

# Save the best model
print(f"\nSaving best model from Fold {best_fold}")
best_model.save_pretrained("./SpanBERT-SM-Large")
tokenizer.save_pretrained("./SpanBERT-SM-Large")
