# Import required libraries
import torch
from torchvision import models, transforms, datasets
from PIL import Image
import os
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import time

matplotlib.rcParams['font.family'] = 'DejaVu Serif'

# Hardware constraints
VRAM_LIMIT_GB = 2.0
NUM_CPU_CORES = 4

# Set CPU cores (software limitation)
torch.set_num_threads(NUM_CPU_CORES)
torch.set_num_interop_threads(NUM_CPU_CORES)

# Define the CT MobileNetV2 class
class CT_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(CT_MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Define data directories
data_dir = './CT_scan_processed_128x128'

# Define the transform to preprocess the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Create dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set VRAM limit
if device.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    fraction = (VRAM_LIMIT_GB * 1024**3) / total_vram
    torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0), device=0)

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

model = CT_MobileNetV2()
model.to(device)

weights_path = './MobileNetV2_CT_Scan/MobileNetV2_fold_1_best.pth'
state_dict = torch.load(weights_path, map_location=device, weights_only=True)

# Adjust state_dict keys to prevent model loading failure
adjusted_state_dict = {}
for key, value in state_dict.items():
    new_key = key
    if key.startswith('classifier.1.1'):
        new_key = key.replace('classifier.1.1', 'model.classifier.1')
    else:
        new_key = f"model.{key}"
    adjusted_state_dict[new_key] = value

# Load the adjusted state_dict into the model
model.load_state_dict(adjusted_state_dict)
model.eval()

# Calculate model size
model_size = get_model_size_mb(model)

# Reset VRAM stats after loading model, before inference
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

class_names = dataset.classes

# Initialize lists to store true labels and predicted scores
true_labels = []
predicted_scores = []
inference_times = []
vram_samples = []

# Iterate through the dataset and make predictions
for i, (inputs, labels) in enumerate(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
        true_class = labels.item()

        # Store true labels and probabilities
        true_labels.append(true_class)
        predicted_scores.append(probabilities)  # Store all class probabilities

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = (time.perf_counter() - start_time) * 1000
    inference_times.append(inference_time)

    # Sample VRAM every 100 iterations
    if device.type == "cuda" and i % 100 == 0:
        vram_samples.append(torch.cuda.memory_allocated() / (1024**2))

true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(class_names)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], np.array(predicted_scores)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), np.array(predicted_scores).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Average & compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Random classifier line for 4 class classification
x = np.linspace(0, 1, 100)
y = x/4  # Equation for random classifier; AUC should be 0.25

# Micro-Average ROC curve
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Plot Micro-Average ROC curve
ax.plot(fpr["micro"], tpr["micro"],
        color='#7B1FA2',  
        linewidth=3.0,
        label=f'CT scan Modality',
        zorder=2)

# Plot random classifier
ax.plot(x, y,
        color='#D32F2F',  
        linewidth=2.0,
        linestyle='--',
        label='Random Classifier',
        zorder=1)

ax.fill_between(fpr["micro"], tpr["micro"], alpha=0.15, color='#7B1FA2', zorder=1)

ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
ax.set_title('Micro ROC Curve: CT scan Modality', fontsize=18, fontweight='bold', pad=20)

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])

ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1.0, length=3)

ax.minorticks_on()

ax.grid(True, alpha=0.35, linestyle='-', linewidth=1.0, color='gray', which='major')
ax.grid(True, alpha=0.20, linestyle='-', linewidth=0.6, color='gray', which='minor')
ax.set_axisbelow(True)

legend = ax.legend(loc='lower right',
                   fontsize=13,
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   framealpha=0.95)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#CCCCCC')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#333333')

ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# Save Micro-Average plot
micro_roc_png_path = 'Micro-Average_ROC_Curve_CT_Scan.png'
plt.savefig(micro_roc_png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# Macro-Average ROC curve
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Plot Macro-Average ROC curve
ax.plot(fpr["macro"], tpr["macro"],
        color='#F57C00',  
        linewidth=3.0,
        label=f'CT scan Modality',
        zorder=2)

# Plot random classifier
ax.plot(x, y,
        color='#D32F2F', 
        linewidth=2.0,
        linestyle='--',
        label='Random Classifier',
        zorder=1)

ax.fill_between(fpr["macro"], tpr["macro"], alpha=0.15, color='#F57C00', zorder=1)

ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
ax.set_title('Macro ROC Curve: CT scan Modality', fontsize=18, fontweight='bold', pad=20)

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])

ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1.0, length=3)

ax.minorticks_on()

ax.grid(True, alpha=0.35, linestyle='-', linewidth=1.0, color='gray', which='major')
ax.grid(True, alpha=0.20, linestyle='-', linewidth=0.6, color='gray', which='minor')
ax.set_axisbelow(True)

legend = ax.legend(loc='lower right',
                   fontsize=13,
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   framealpha=0.95)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#CCCCCC')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#333333')

ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# Save Macro-Average plot
macro_roc_png_path = 'Macro-Average_ROC_Curve_CT_Scan.png'
plt.savefig(macro_roc_png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Micro-Averaged ROC curve saved to {micro_roc_png_path}")
print(f"Macro-Averaged ROC curve saved to {macro_roc_png_path}")

# Calculate hardware efficiency metrics
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

# Print efficiency metrics
print("\nHardware Efficiency Metrics")
print("-"*50)
print(f"Hardware Constraints: {NUM_CPU_CORES} CPU cores, {VRAM_LIMIT_GB} GB VRAM")
print(f"Inference Time (ms):     {inference_time_mean:.3f} ± {inference_time_std:.3f}")
print(f"VRAM Usage (MB):         {vram_mean:.2f} ± {vram_std:.2f}")
print(f"Peak VRAM (MB):          {peak_vram:.2f}")
print(f"Model Size (MB):         {model_size:.2f}")
print(f"Throughput (samples/s):  {throughput:.2f}")