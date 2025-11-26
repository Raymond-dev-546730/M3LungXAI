# Import required libraries
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from torchvision import datasets, transforms, models
import pandas as pd

# Set random seed for reproducibility (42)
np.random.seed(42)
torch.manual_seed(42)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the best saved epoch states
model_paths = {
    "ResNet18": "./ResNet18_CXR14/epoch_21_model.pth",
    "ConvNeXtTiny": "./ConvNeXtTiny_CXR14/epoch_9_model.pth",
    "EfficientNetV2S": "./EfficientNetV2S_CXR14/epoch_8_model.pth", 
    "DenseNet121": "./DenseNet121_CXR14/epoch_10_model.pth" 
}

# Load full dataset with validation transforms
data_dir = "./CXR14_processed_224x224" 
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
full_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

# Extract labels for stratification
y_all = np.array([label for _, label in full_dataset.imgs])

# Split data into train+val and test sets
train_val_idx, test_idx = train_test_split(
    np.arange(len(y_all)),
    test_size=0.1,
    stratify=y_all,
    random_state=42
)

# Function to load model based on name
def load_model(model_name):
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "ConvNeXtTiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    elif model_name == "EfficientNetV2S":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError("FATAL ERROR. MODEL WEIGHTS NOT PRESENT.")
    
    model.load_state_dict(torch.load(model_paths[model_name], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

# Function to get predictions from a model for specific indices
def get_predictions_for_indices(model, dataset, indices, batch_size=32):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probabilities.cpu().numpy())
    return np.array(all_preds)

# Function to optimize threshold for F1 score
def optimize_threshold(y_true, y_pred_probs):
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_f1, best_t = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_pred_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

if __name__ == '__main__':
    # Load all 4 CNN models
    print("Loading CNN models...")
    resnet18 = load_model("ResNet18")
    convnexttiny = load_model("ConvNeXtTiny")
    efficientnetv2s = load_model("EfficientNetV2S")
    densenet121 = load_model("DenseNet121")

    # Create output directory
    output_dir = "./Averaged_Meta-Models-Ensemble"
    os.makedirs(output_dir, exist_ok=True)

    print("Top-3 Models: GB Base, MLP Base, GB Engineered")
    print("Method: Simple average of OOF probabilities")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for OOF predictions from top-3 models
    oof_gb_base = np.zeros(len(train_val_idx))
    oof_mlp_base = np.zeros(len(train_val_idx))
    oof_gb_eng = np.zeros(len(train_val_idx))
    
    all_fold_metrics = {
        'auc': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []
    }

    for fold, (train_idx_fold, val_idx_fold) in enumerate(skf.split(train_val_idx, y_all[train_val_idx])):
        print(f'\nFold {fold + 1}/5')
        print('-' * 50)
        
        train_indices = train_val_idx[train_idx_fold]
        val_indices = train_val_idx[val_idx_fold]
        
        y_train_fold = y_all[train_indices]
        y_val_fold = y_all[val_indices]
        
        # Get CNN predictions on training set
        print("Getting CNN predictions on training set...")
        resnet18_train = get_predictions_for_indices(resnet18, full_dataset, train_indices)
        convnexttiny_train = get_predictions_for_indices(convnexttiny, full_dataset, train_indices)
        efficientnetv2s_train = get_predictions_for_indices(efficientnetv2s, full_dataset, train_indices)
        densenet121_train = get_predictions_for_indices(densenet121, full_dataset, train_indices)
        
        X_train = np.column_stack([resnet18_train, convnexttiny_train, efficientnetv2s_train, densenet121_train])
        
        # Get CNN predictions on validation set
        print("Getting CNN predictions on validation set...")
        resnet18_val = get_predictions_for_indices(resnet18, full_dataset, val_indices)
        convnexttiny_val = get_predictions_for_indices(convnexttiny, full_dataset, val_indices)
        efficientnetv2s_val = get_predictions_for_indices(efficientnetv2s, full_dataset, val_indices)
        densenet121_val = get_predictions_for_indices(densenet121, full_dataset, val_indices)
        
        X_val = np.column_stack([resnet18_val, convnexttiny_val, efficientnetv2s_val, densenet121_val])
        
        # Train top-3 models and generate OOF predictions
        print("\nTraining Model 1/3: Gradient Boosting (Base)")
        gb_base = GradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=2, n_estimators=100,
            subsample=0.8, random_state=42
        )
        gb_base.fit(X_train, y_train_fold)
        oof_gb_base_fold = gb_base.predict_proba(X_val)[:, 1]
        
        print("Training Model 2/3: MLP (Base)")
        mlp_base = MLPClassifier(
            activation='relu', alpha=0.0001, hidden_layer_sizes=(64, 32),
            learning_rate_init=0.01, random_state=42, max_iter=500,
            early_stopping=True, validation_fraction=0.1
        )
        mlp_base.fit(X_train, y_train_fold)
        oof_mlp_base_fold = mlp_base.predict_proba(X_val)[:, 1]
        
        print("Training Model 3/3: Gradient Boosting (Engineered)")
        # Engineer features for GB Engineered
        def engineer_features(cnn_probs):
            n_samples = cnn_probs.shape[0]
            features = np.zeros((n_samples, 11))
            features[:, 0:4] = cnn_probs
            features[:, 4] = np.mean(cnn_probs, axis=1)
            features[:, 5] = np.var(cnn_probs, axis=1)
            features[:, 6] = np.min(cnn_probs, axis=1)
            features[:, 7] = np.max(cnn_probs, axis=1)
            features[:, 8] = features[:, 7] - features[:, 6]
            features[:, 9] = np.sum(cnn_probs >= 0.5, axis=1)
            epsilon = 1e-10
            p_safe = np.clip(cnn_probs, epsilon, 1 - epsilon)
            entropies = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))
            features[:, 10] = np.mean(entropies, axis=1)
            return features
        
        X_train_eng = engineer_features(X_train)
        X_val_eng = engineer_features(X_val)
        
        gb_eng = GradientBoostingClassifier(
            learning_rate=0.03, max_depth=3, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=5, n_estimators=300,
            subsample=1.0, random_state=42
        )
        gb_eng.fit(X_train_eng, y_train_fold)
        oof_gb_eng_fold = gb_eng.predict_proba(X_val_eng)[:, 1]
        
        # Store OOF predictions (map val_indices back to original train_val_idx positions)
        pos = {idx:i for i, idx in enumerate(train_val_idx)}
        val_pos = np.array([pos[idx] for idx in val_indices])
        oof_gb_base[val_pos] = oof_gb_base_fold
        oof_mlp_base[val_pos] = oof_mlp_base_fold
        oof_gb_eng[val_pos]  = oof_gb_eng_fold
        
        # Average the 3 OOF predictions
        avg_oof_val = (oof_gb_base_fold + oof_mlp_base_fold + oof_gb_eng_fold) / 3.0
        
        # Optimize threshold on averaged OOF
        optimal_threshold, train_f1 = optimize_threshold(y_val_fold, avg_oof_val)
        predicted_labels = (avg_oof_val >= optimal_threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_val_fold, avg_oof_val)
        f1 = f1_score(y_val_fold, predicted_labels, average='macro')
        precision = precision_score(y_val_fold, predicted_labels, average='macro', zero_division=1)
        recall = recall_score(y_val_fold, predicted_labels, average='macro')
        accuracy = accuracy_score(y_val_fold, predicted_labels)
        
        all_fold_metrics['auc'].append(auc)
        all_fold_metrics['f1'].append(f1)
        all_fold_metrics['precision'].append(precision)
        all_fold_metrics['recall'].append(recall)
        all_fold_metrics['accuracy'].append(accuracy)
        
        print(f'\nFold {fold + 1} Metrics:')
        print(f'AUC: {auc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Optimal Threshold: {optimal_threshold:.3f}')

    print("Average Cross-Validation Results:")
    print("-"*50)
    print(f"Average AUC: {np.mean(all_fold_metrics['auc']):.4f} ± {np.std(all_fold_metrics['auc']):.4f}")
    print(f"Average F1 Score: {np.mean(all_fold_metrics['f1']):.4f} ± {np.std(all_fold_metrics['f1']):.4f}")
    print(f"Average Precision: {np.mean(all_fold_metrics['precision']):.4f} ± {np.std(all_fold_metrics['precision']):.4f}")
    print(f"Average Recall: {np.mean(all_fold_metrics['recall']):.4f} ± {np.std(all_fold_metrics['recall']):.4f}")
    print(f"Average Accuracy: {np.mean(all_fold_metrics['accuracy']):.4f} ± {np.std(all_fold_metrics['accuracy']):.4f}")
    
    # Save CV metrics
    cv_metrics_df = pd.DataFrame(all_fold_metrics)
    cv_metrics_df.to_csv(os.path.join(output_dir, 'cv_metrics.csv'), index=False)
    
    # Train final models on full train+val
    print("Training final models on full train+val set...")
    
    resnet18_trainval = get_predictions_for_indices(resnet18, full_dataset, train_val_idx)
    convnexttiny_trainval = get_predictions_for_indices(convnexttiny, full_dataset, train_val_idx)
    efficientnetv2s_trainval = get_predictions_for_indices(efficientnetv2s, full_dataset, train_val_idx)
    densenet121_trainval = get_predictions_for_indices(densenet121, full_dataset, train_val_idx)
    
    X_trainval = np.column_stack([resnet18_trainval, convnexttiny_trainval, efficientnetv2s_trainval, densenet121_trainval])
    y_trainval = y_all[train_val_idx]
    
    # Train final top-3 models
    final_gb_base = GradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=2, n_estimators=100,
        subsample=0.8, random_state=42
    )
    final_gb_base.fit(X_trainval, y_trainval)
    
    final_mlp_base = MLPClassifier(
        activation='relu', alpha=0.0001, hidden_layer_sizes=(64, 32),
        learning_rate_init=0.01, random_state=42, max_iter=500,
        early_stopping=True, validation_fraction=0.1
    )
    final_mlp_base.fit(X_trainval, y_trainval)
    
    X_trainval_eng = engineer_features(X_trainval)
    final_gb_eng = GradientBoostingClassifier(
        learning_rate=0.03, max_depth=3, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=5, n_estimators=300,
        subsample=1.0, random_state=42
    )
    final_gb_eng.fit(X_trainval_eng, y_trainval)
    
    # Optimize threshold on full train+val OOF average
    avg_oof_trainval = (oof_gb_base + oof_mlp_base + oof_gb_eng) / 3.0
    final_threshold, _ = optimize_threshold(y_trainval, avg_oof_trainval)
    print(f"Final optimal threshold: {final_threshold:.3f}")
    
    # Predict on test set
    y_test = y_all[test_idx]
    
    resnet18_test = get_predictions_for_indices(resnet18, full_dataset, test_idx)
    convnexttiny_test = get_predictions_for_indices(convnexttiny, full_dataset, test_idx)
    efficientnetv2s_test = get_predictions_for_indices(efficientnetv2s, full_dataset, test_idx)
    densenet121_test = get_predictions_for_indices(densenet121, full_dataset, test_idx)
    
    X_test = np.column_stack([resnet18_test, convnexttiny_test, efficientnetv2s_test, densenet121_test])
    
    test_gb_base_probs = final_gb_base.predict_proba(X_test)[:, 1]
    test_mlp_base_probs = final_mlp_base.predict_proba(X_test)[:, 1]
    
    X_test_eng = engineer_features(X_test)
    test_gb_eng_probs = final_gb_eng.predict_proba(X_test_eng)[:, 1]
    
    # Average test predictions
    test_avg_probs = (test_gb_base_probs + test_mlp_base_probs + test_gb_eng_probs) / 3.0
    test_preds = (test_avg_probs >= final_threshold).astype(int)
    
    # Calculate test metrics
    test_auc = roc_auc_score(y_test, test_avg_probs)
    test_f1 = f1_score(y_test, test_preds, average='macro')
    test_precision = precision_score(y_test, test_preds, average='macro', zero_division=1)
    test_recall = recall_score(y_test, test_preds, average='macro')
    test_accuracy = accuracy_score(y_test, test_preds)
    
    test_metrics = {
        'auc': test_auc,
        'f1': test_f1,
        'precision': test_precision,
        'recall': test_recall,
        'accuracy': test_accuracy
    }
    
    # Save test metrics
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)
    
    print(f'\nTest AUC: {test_auc:.4f}')
    print(f'Test F1: {test_f1:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

