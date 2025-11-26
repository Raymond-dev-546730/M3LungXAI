# Import required libraries
import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2

def enhance_sharpness(image, factor=1.2):  
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(factor)
    return image

def increase_contrast(image, factor=1.2):  
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor)
    return image

def apply_unsharp_mask(image, radius=1, percent=100, threshold=5):  
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def apply_edge_enhancement(image):
    return image.filter(ImageFilter.EDGE_ENHANCE)  

def apply_histogram_equalization(image):
    image_np = np.array(image)
    if image.mode != 'L':
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.equalizeHist(image_np))

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))  
    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale
    image_np = np.array(image)
    image_np = clahe.apply(image_np.astype(np.uint8))  # Ensure the image is uint8 type
    return Image.fromarray(image_np)

# Function to resize image with padding
def resize_with_padding(image, target_size):
    old_size = image.size  
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.BICUBIC)
    new_image = Image.new("L", (target_size, target_size))
    new_image.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))

    # Add a border if there's too much padding
    if (target_size - new_size[0]) > 0.2 * target_size or (target_size - new_size[1]) > 0.2 * target_size:
        new_image = ImageOps.expand(new_image, border=10, fill='black')
    
    return new_image

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    
    # Apply unsharp mask for deblurring 
    img = apply_unsharp_mask(img, radius=1, percent=100, threshold=5)
    
    # Apply edge enhancement 
    img = apply_edge_enhancement(img)
    
    # Apply sharpness enhancement 
    img = enhance_sharpness(img)
    
    # Increase contrast 
    img = increase_contrast(img)
    
    # Apply histogram equalization
    img = apply_histogram_equalization(img)
    
    # Resize with padding using bicubic interpolation
    img = resize_with_padding(img, 224)
    
    # Apply CLAHE 
    img = apply_clahe(img)
    
    return img

# Function to save the processed image
def save_image(image, output_folder, output_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    save_path = os.path.join(output_folder, output_name)
    
    # Save the image
    image.save(save_path)

def process():
    # Input folder and output paths
    input_folder = './Pre_Input_X-ray'
    os.makedirs(input_folder, exist_ok=True)
    output_folder = './Input_X-ray'
    output_image_name = 'processed_image.png'
    
    # Get the first image file in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp'))]
    
    if not image_files:
        print("FATAL ERROR. No image files found in the folder. Preferred image type is png.")
        return
    
    input_image_path = os.path.join(input_folder, image_files[0])
    
    
    # Load and preprocess the image
    processed_image = load_and_preprocess_image(input_image_path)
    
    # Save the preprocessed image
    save_image(processed_image, output_folder, output_image_name)

if __name__ == '__main__':
    process()


