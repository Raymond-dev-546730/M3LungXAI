# Import required libraries
import json
import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from lime import lime_image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from lime.wrappers.scikit_image import SegmentationAlgorithm
import random
import cv2

# Set random seed for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the best saved epoch states (hand picked based on validation AUC and Loss)
model_paths = {
    "ResNet18": "./X-ray_Modality/ResNet18_CXR14/epoch_21_model.pth",
    "ConvNeXtTiny": "./X-ray_Modality/ConvNeXtTiny_CXR14/epoch_9_model.pth",
    "EfficientNetV2S": "./X-ray_Modality/EfficientNetV2S_CXR14/epoch_8_model.pth", 
    "DenseNet121": "./X-ray_Modality/DenseNet121_CXR14/epoch_10_model.pth" 
}

# Path for the GB ensemble model and config
gb_model_path = "./X-ray_Modality/Ensemble_Model_CXR14/GB_Model.joblib"
model_config_path = "./X-ray_Modality/Ensemble_Model_CXR14/Model_Config.json"

# Load the configuration
with open(model_config_path, 'r') as f:
    config = json.load(f)
    gb_threshold = config['gb_threshold']
    feature_order = config['feature_order']

# Transformations for the input image
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

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
    
    model.load_state_dict(torch.load(model_paths[model_name], map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Load models once in correct order
cnn_models = {name: load_model(name) for name in feature_order}

def predict_cnn_features(image_tensor):
    cnn_features = []
    for name in feature_order:
        with torch.no_grad():
            output = cnn_models[name](image_tensor.unsqueeze(0).to(device))
            probability = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            cnn_features.append(probability[0])
    return np.array(cnn_features).reshape(1, -1)

def plot_cnn_influence(cnn_features, prediction_label, output_dir):
    cnn_influences = {
        name: cnn_features[0, idx] 
        for idx, name in enumerate(feature_order)
    }

    plt.figure(figsize=(10, 6))
    plt.bar(cnn_influences.keys(), cnn_influences.values(), color='#4A90E2')
    plt.xlabel('CNN Models')
    plt.ylabel(f'Probability for Class "{prediction_label}"')
    plt.title(f'CNN Model Influence for Class "{prediction_label}"')
    plt.xticks(rotation=25)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'X-ray_CNN_Influence.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

# Global variable for Streamlit integration
prediction_results = {}

def predict():

    global prediction_results
    
    # Load and preprocess image
    input_dir = "./Input_X-ray" 
    os.makedirs(input_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if len(image_files) != 1:
        raise ValueError(f"Expected exactly one image in '{input_dir}', but found {len(image_files)}.")
    
    image_path = os.path.join(input_dir, image_files[0])
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transforms(image)
    
    # Get CNN feature predictions
    cnn_features = predict_cnn_features(image_tensor)
    
    # Load the GB ensemble model
    gb_model = joblib.load(gb_model_path)

    # Final ensemble prediction
    final_pred_prob = gb_model.predict_proba(cnn_features)[:, 1][0]
    final_pred = (final_pred_prob >= gb_threshold).astype(int)
    prediction_label = 'Normal' if final_pred == 1 else 'Nodule'
    target_label = 1 if final_pred == 1 else 0

    print(f"[DEBUG] PREDICTION: {prediction_label}")
    
    # Get confidence for the predicted class
    probabilities = gb_model.predict_proba(cnn_features)[0]
    
    print(f"[DEBUG] CONFIDENCE: {probabilities[target_label]*100:.2f}%")

    # Store results for Streamlit integration
    prediction_results = {
        'xray_prediction': prediction_label,
        'xray_confidence': probabilities[target_label]*100
    }

    # Plot and save CNN influence chart
    output_dir = "./XAI_Output_1"
    os.makedirs(output_dir, exist_ok=True)
    plot_cnn_influence(cnn_features, prediction_label, output_dir)

    # Define LIME explainer
    explainer = lime_image.LimeImageExplainer(random_state=1)

    # LIME prediction function
    def gb_predict(images):
        preds = []
        for img in images:
            img_tensor = val_transforms(Image.fromarray(img)).to(device)
            cnn_feats = predict_cnn_features(img_tensor)
            probas = gb_model.predict_proba(cnn_feats)[0]
            preds.append(probas)
        return np.array(preds)

    segmentation_fn = SegmentationAlgorithm(
        'slic',
        n_segments=50,
        compactness=20,
        sigma=1,
        random_seed=1
    )

    explanation = explainer.explain_instance(
        np.array(image),
        gb_predict,
        labels=[0, 1],
        hide_color=0,
        num_samples=100,
        segmentation_fn=segmentation_fn
    )

    if target_label in explanation.top_labels:
        class_name = "Nodule" if target_label == 0 else "Normal"
        temp, mask = explanation.get_image_and_mask(
            label=target_label,
            positive_only=False,
            num_features=5,
            hide_rest=False
        )

        exp_list = explanation.local_exp[target_label]
        exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)

        seg_ids = [str(x[0]) for x in exp_list]
        seg_weights = [x[1] for x in exp_list]
        bar_colors = ["green" if w > 0 else "red" for w in seg_weights]

        plt.figure(figsize=(8, 6))
        plt.bar(seg_ids, seg_weights, color=bar_colors)
        plt.title(f"LIME Local Feature Importances for '{class_name}'")
        plt.xlabel("Super-Pixel ID")
        plt.ylabel("Importance Weight")
        plt.xticks(rotation=45, fontsize=8)
        lime_bar_path = os.path.join(output_dir, "X-ray_LIME_Bar.png")
        plt.savefig(lime_bar_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot top 5 features
        k = 5
        top_segments = exp_list[:k]
        segments = explanation.segments

        lime_overlay = temp.astype(np.float32)
        alpha = 0.4
        for (seg_id, weight) in top_segments:
            mask_area = (segments == seg_id)
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)
            lime_overlay[mask_area] = alpha * lime_overlay[mask_area] + (1 - alpha) * np.array(color, dtype=np.float32)
            y_indices, x_indices = np.where(mask_area)
            if len(y_indices) > 0:
                y_mean = int(np.mean(y_indices))
                x_mean = int(np.mean(x_indices))
                label_text = f"ID:{seg_id}"
                cv2.putText(lime_overlay, label_text, (x_mean, y_mean), 
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.3, 
                            color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        lime_overlay = np.clip(lime_overlay, 0, 255).astype(np.uint8)
        overlay_path = os.path.join(output_dir, "X-ray_LIME_Overlay.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(lime_overlay)
        plt.axis('off')
        plt.title(f"LIME Overlay (Top {k} Super-Pixels) for '{class_name}'")
        plt.savefig(overlay_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        print(f"LIME did not generate an explanation for the predicted class '{prediction_label}'.")

if __name__ == '__main__':
    predict()