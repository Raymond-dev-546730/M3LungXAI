# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader

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

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

# Dataset preparation
class spanned_dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

# Load model and tokenizer
model_path = "./SpanBERT-SM-Large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Calculate model size
model_size = get_model_size_mb(model)

# Reset VRAM stats after loading model, before inference
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Load full dataset
jsonl_path = "dataset_index.jsonl"
dataset = spanned_dataset(jsonl_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Inference with tracking
y_true = []
y_scores = []
inference_times = []
vram_samples = []

total_samples = len(dataset)

for idx, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].long().to(device)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        # Get probabilities for positive class (SYMPTOM) at token level
        prob_positive = probs[:, :, 1].view(-1).cpu().numpy()
        true_labels = labels.view(-1).cpu().numpy()
        mask = attention_mask.view(-1).cpu().numpy() == 1
        
        # Filter out padding tokens - token-level evaluation
        y_true.extend(true_labels[mask])
        y_scores.extend(prob_positive[mask])
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = (time.perf_counter() - start_time) * 1000
    inference_times.append(inference_time)
    
    # Sample VRAM every 10 iterations
    if device.type == "cuda" and idx % 10 == 0:
        vram_samples.append(torch.cuda.memory_allocated() / (1024**2))
    
    if (idx + 1) % 100 == 0:
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

# Plot ROC design 
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

ax.plot(fpr, tpr, color='#2E7D32', linewidth=3.0, label=f'Symptom Modality', zorder=2)
ax.plot([0, 1], [0, 1], color='#D32F2F', linewidth=2.0, linestyle='--', label='Random Classifier', zorder=1)
ax.fill_between(fpr, tpr, alpha=0.15, color='#2E7D32', zorder=1)

ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
ax.set_title('ROC Curve: Symptom Modality', fontsize=18, fontweight='bold', pad=20)
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
plt.savefig('ROC_Curve_Symptom.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print efficiency metrics
print("\nHardware Efficiency Metrics")
print("-"*50)
print(f"Hardware Constraints: {NUM_CPU_CORES} CPU cores, {VRAM_LIMIT_GB} GB VRAM")
print(f"Inference Time (ms):     {inference_time_mean:.3f} ± {inference_time_std:.3f}")
print(f"VRAM Usage (MB):         {vram_mean:.2f} ± {vram_std:.2f}")
print(f"Peak VRAM (MB):          {peak_vram:.2f}")
print(f"Model Size (MB):         {model_size:.2f}")
print(f"Throughput (samples/s):  {throughput:.2f}")