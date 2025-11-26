# Import required libraries
import torch
from torchvision import models, transforms, datasets 
import torch.nn as nn
import numpy as np
import joblib
import json
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

matplotlib.rcParams['font.family'] = 'DejaVu Serif'

# Hardware constraints
VRAM_LIMIT_GB = 2.0
NUM_CPU_CORES = 4

# Set CPU cores (software limitation)
torch.set_num_threads(NUM_CPU_CORES)
torch.set_num_interop_threads(NUM_CPU_CORES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set VRAM limit
if device.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    fraction = (VRAM_LIMIT_GB * 1024**3) / total_vram
    torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0), device=0)

model_paths = {
    "ResNet18": "./ResNet18_CXR14/epoch_21_model.pth",
    "ConvNeXtTiny": "./ConvNeXtTiny_CXR14/epoch_9_model.pth",
    "EfficientNetV2S": "./EfficientNetV2S_CXR14/epoch_8_model.pth", 
    "DenseNet121": "./DenseNet121_CXR14/epoch_10_model.pth"
}

ensemble_model_path = "./Ensemble_Model_CXR14/GB_Model.joblib"
config_path = "./Ensemble_Model_CXR14/Model_Config.json"

# Load config
with open(config_path, 'r') as f:
    config = json.load(f)
    feature_order = config['feature_order']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 ** 2)

def load_model(model_name):
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif model_name == "ConvNeXtTiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    elif model_name == "EfficientNetV2S":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError("FATAL ERROR. MODEL WEIGHTS NOT PRESENT.")
    
    state_dict = torch.load(model_paths[model_name], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Load models in correct feature order
cnn_models = {}
total_model_size = 0

for name in feature_order:
    model = load_model(name)
    cnn_models[name] = model
    total_model_size += get_model_size_mb(model)

ensemble_model = joblib.load(ensemble_model_path)
total_model_size += get_file_size_mb(ensemble_model_path)

# Reset VRAM stats after loading models, before inference
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Load dataset
data_dir = './CXR14_processed_224x224'
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Inference with tracking
y_true = []
y_scores = []
inference_times = []
vram_samples = []

total_samples = len(dataset)

for idx, (inputs, labels) in enumerate(dataloader):
    inputs = inputs.to(device)
    y_true.append(labels.item())
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    cnn_features = []
    with torch.no_grad():
        for name in feature_order:
            model = cnn_models[name]
            output = model(inputs)
            probability = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            cnn_features.append(probability[0])
    
    X = np.array(cnn_features).reshape(1, -1)
    final_prob = ensemble_model.predict_proba(X)[0, 1]
    y_scores.append(final_prob)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = (time.perf_counter() - start_time) * 1000
    inference_times.append(inference_time)
    
    # Sample VRAM every 100 iterations
    if device.type == "cuda" and idx % 100 == 0:
        vram_samples.append(torch.cuda.memory_allocated() / (1024**2))
    
    if (idx + 1) % 10000 == 0:
        print(f"Progress: {idx+1}/{total_samples}")

y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Calculate metrics
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

inference_time_mean = np.mean(inference_times)
inference_time_std = np.std(inference_times)
throughput = 1000 / inference_time_mean

if device.type == "cuda":
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    vram_mean = np.mean(vram_samples)
    vram_std = np.std(vram_samples)
else:
    peak_vram = 0
    vram_mean = 0
    vram_std = 0

# Plot ROC
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

ax.plot(fpr, tpr, color='#2E86AB', linewidth=3.0, label=f'X-ray Modality', zorder=2)
ax.plot([0, 1], [0, 1], color='#D32F2F', linewidth=2.0, linestyle='--', label='Random Classifier', zorder=1)
ax.fill_between(fpr, tpr, alpha=0.15, color='#2E86AB', zorder=1)

ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
ax.set_title('ROC Curve: X-ray Modality', fontsize=18, fontweight='bold', pad=20)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1.0, length=3)
ax.minorticks_on()

ax.grid(True, alpha=0.35, linestyle='-', linewidth=1.0, color='gray', which='major')
ax.grid(True, alpha=0.20, linestyle='-', linewidth=0.6, color='gray', which='minor')
ax.set_axisbelow(True)

legend = ax.legend(loc='lower right', fontsize=13, frameon=True, fancybox=True, shadow=True, framealpha=0.95)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#CCCCCC')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#333333')

ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('ROC_Curve_X-ray.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print efficiency metrics
print("\nHardware Efficiency Metrics")
print("-"*50)
print(f"Hardware Constraints: {NUM_CPU_CORES} CPU cores, {VRAM_LIMIT_GB} GB VRAM")
print(f"Inference Time (ms):     {inference_time_mean:.3f} ± {inference_time_std:.3f}")
print(f"VRAM Usage (MB):         {vram_mean:.2f} ± {vram_std:.2f}")
print(f"Peak VRAM (MB):          {peak_vram:.2f}")
print(f"Model Size (MB):         {total_model_size:.2f}")
print(f"Throughput (samples/s):  {throughput:.2f}")
