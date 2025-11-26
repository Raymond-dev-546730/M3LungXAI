# Import required libraries
import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2

# Set random seed for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# Define the CT MobileNetV2 class
class CT_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(CT_MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CT_MobileNetV2().to(device)
weights_path = './CT_Scan_Modality/MobileNetV2_CT_Scan/MobileNetV2_fold_1_best.pth'
state_dict = torch.load(weights_path, map_location=device, weights_only=True)

# Adjust state_dict keys to prevent model loading failure
adjusted_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('classifier.1.1'):
        new_key = key.replace('classifier.1.1', 'model.classifier.1')
    else:
        new_key = f"model.{key}"
    adjusted_state_dict[new_key] = value

model.load_state_dict(adjusted_state_dict)
model.eval()

prediction_results = {} 

def predict():
    global prediction_results
    input_dir = './Input_CT-Scan'
    os.makedirs('./Input_CT-Scan', exist_ok=True) 
    input_files = os.listdir(input_dir)

    if len(input_files) == 0:
        print("No images found in the Input_CT-Scan directory.")
        print("Please place exactly one image in the 'Input_CT-Scan' folder and re-run.")
        return
    elif len(input_files) > 1:
        print("Error: More than one file found in the Input_CT-Scan directory.")
        print("Please ensure there is exactly one image file in the 'Input_CT-Scan' folder and re-run.")
        return

    img_file = input_files[0]
    img_path = os.path.join(input_dir, img_file)

    # Load & Convert to RGB
    image_pil = Image.open(img_path).convert('RGB')
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

    predicted_class = np.argmax(probabilities)
    confidence_score = probabilities[predicted_class] * 100
    print(f"[DEBUG] PREDICTION: {class_names[predicted_class]}") # Prints out predicted class
    print(f"[DEBUG] CONFIDENCE: {confidence_score:.2f}%") # Prints out confidence score

    prediction_results = {
        'ct_prediction': class_names[predicted_class],
        'ct_confidence': confidence_score
    }

    def target_class_function(out):
        return out[predicted_class]

    target_layer = model.model.features[-1]
    cam = GradCAMPlusPlus(model, target_layer)

    grayscale_cam = cam(
        input_tensor,
        targets=[target_class_function],
        aug_smooth=True,
        eigen_smooth=True
    )
    grayscale_cam = grayscale_cam[0, :]

    # Overlay CAM on original
    original_image = np.array(image_pil, dtype=np.uint8)
    original_image_float = np.float32(original_image) / 255.0

    visualization = show_cam_on_image(
        original_image_float, 
        grayscale_cam, 
        use_rgb=True
    )

    # Save Grad-CAM++ result
    os.makedirs('./XAI_Output_2', exist_ok=True)
    gradcam_path = os.path.join('./XAI_Output_2', "CT_GradCAM++.png")
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    gradcam_pil = Image.fromarray(cv2.cvtColor(visualization_bgr, cv2.COLOR_BGR2RGB))
    gradcam_pil.save(gradcam_path, dpi=(300, 300))

    def classifier_fn(images):
        input_batch = []
        for img_array in images:
            pil_img = Image.fromarray((img_array*255).astype(np.uint8))
            tensor_img = transform(pil_img).unsqueeze(0).to(device)
            input_batch.append(tensor_img)

        # Combine all into one batch
        input_batch = torch.cat(input_batch, dim=0)

        with torch.no_grad():
            output_logits = model(input_batch)
            probs = torch.nn.functional.softmax(output_logits, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer(random_state=1)

    image_np = np.array(image_pil, dtype=np.float32) / 255.0

    # SLIC is automatically used by default
    explanation = explainer.explain_instance(
        image=image_np,
        classifier_fn=classifier_fn,
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )

    exp_list = explanation.local_exp[predicted_class]  
    # Sort by descending absolute weight so the largest influences show first
    exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)

    seg_ids = [str(x[0]) for x in exp_list]
    seg_weights = [x[1] for x in exp_list]
    bar_colors = ["green" if w > 0 else "red" for w in seg_weights]

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(seg_ids, seg_weights, color=bar_colors)
    plt.title(f"LIME Local Feature Importances for '{class_names[predicted_class]}'")
    plt.xlabel("Super-Pixel ID")
    plt.ylabel("Importance Weight")

    lime_bar_path = os.path.join('./XAI_Output_2', "CT_LIME_Bar.png")
    plt.savefig(lime_bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Only plots the top 5 most important features
    k = 5
    top_segments = exp_list[:k] 

    segments = explanation.segments  
    lime_overlay = (image_np * 255).astype(np.uint8).copy()
    alpha = 0.4

    for (seg_id, weight) in top_segments:
        mask = (segments == seg_id)
        color = (0, 255, 0) if weight > 0 else (255, 0, 0)
        lime_overlay[mask] = alpha * lime_overlay[mask] + (1 - alpha) * np.array(color, dtype=np.float32)

        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            y_mean = int(np.mean(y_indices))
            x_mean = int(np.mean(x_indices))
            label_text = f"ID:{seg_id}"
            cv2.putText(
                lime_overlay,
                label_text,
                (x_mean, y_mean),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.3,        
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )

    # Convert to BGR for saving
    lime_overlay_bgr = cv2.cvtColor(lime_overlay, cv2.COLOR_RGB2BGR)
    lime_overlay_path = os.path.join('./XAI_Output_2', "CT_LIME_Overlay.png")
    lime_overlay_pil = Image.fromarray(cv2.cvtColor(lime_overlay_bgr, cv2.COLOR_BGR2RGB))
    lime_overlay_pil.save(lime_overlay_path, dpi=(300, 300))

if __name__ == '__main__':
    predict()