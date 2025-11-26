# Import required libraries
import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2

# Enhance the sharpness 
def enhance_sharpness(image, factor=1.2):
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(factor)
    return image

# Increase the contrast 
def increase_contrast(image, factor=1.2):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor)
    return image

# Apply unsharp masking to enhance image details and reduce blur
def apply_unsharp_mask(image, radius=1, percent=100, threshold=5):
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

# Enhance edges in the image 
def apply_edge_enhancement(image):
    return image.filter(ImageFilter.EDGE_ENHANCE)

# Apply histogram equalization to improve image contrast
def apply_histogram_equalization(image):
    image_np = np.array(image)
    if image.mode != 'L':
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.equalizeHist(image_np))

# Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale
    image_np = np.array(image)
    image_np = clahe.apply(image_np.astype(np.uint8))
    return Image.fromarray(image_np)

# Resize image to target size while maintaining aspect ratio using padding
def resize_with_padding(image, target_size):
    old_size = image.size
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.BICUBIC)
    new_image = Image.new("L", (target_size, target_size))
    new_image.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))

    # Add a border if padding exceeds 20% of target size
    if (target_size - new_size[0]) > 0.2 * target_size or (target_size - new_size[1]) > 0.2 * target_size:
        new_image = ImageOps.expand(new_image, border=10, fill='black')
    
    return new_image

# Save processed images to appropriate folders based on their labels
def save_images(images, labels, output_folder, prefix="image"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, img in enumerate(images):
        label = labels[i]
        if label == 1:
            label_folder = os.path.join(output_folder, 'nodule')
        else:
            label_folder = os.path.join(output_folder, 'normal')
        
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        save_path = os.path.join(label_folder, f"{prefix}_{i}.png")
        
        # Convert normalized values back to uint8 range
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).convert('L').save(save_path)

# Load and apply preprocessing pipeline to medical images
def load_and_preprocess_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        
        # Apply preprocessing pipeline
        img = apply_unsharp_mask(img, radius=1, percent=100, threshold=5)
        img = apply_edge_enhancement(img)
        img = enhance_sharpness(img)
        img = increase_contrast(img)
        img = apply_histogram_equalization(img)
        img = resize_with_padding(img, 224)
        img = apply_clahe(img)
        
        # Normalize pixel values to [0,1] range
        img_np = np.array(img).astype(np.float32) / 255.0
        
        images.append(img_np)
        labels.append(label)
    return images, labels

def process():
    base_path = './CXR14'
    nodule_path = os.path.join(base_path, 'nodule')
    normal_path = os.path.join(base_path, 'normal')

    # Load and preprocess both image classes
    nodule_images, nodule_labels = load_and_preprocess_images(nodule_path, 1)
    normal_images, normal_labels = load_and_preprocess_images(normal_path, 0)

    # Combine datasets
    images = nodule_images + normal_images
    labels = nodule_labels + normal_labels

    # Define output directory for processed images
    output_path = './CXR14_processed_224x224'

    # Save processed images
    save_images(images, labels, output_path, prefix="image")

if __name__ == '__main__':
    process()