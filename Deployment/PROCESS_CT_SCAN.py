# Import required libraries
import os
import cv2
import numpy as np

def load_and_preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("FATAL ERROR. No image files found in the folder.")
    
    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to 128x128
    img = cv2.resize(img, (128, 128))
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    return img

def save_image(image, output_folder, output_image_name):

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, output_image_name)
    
    # Convert image from float to uint8 
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Convert image from RGB back to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, image_bgr)

def process():
    # Input folder and output paths
    input_folder = './Pre_Input_CT-Scan'
    os.makedirs(input_folder, exist_ok=True)
    output_folder = './Input_CT-Scan'
    output_image_name = 'processed_image.png'
    
    # Get the first image file in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp'))]
    
    if not image_files:
        print("FATAL ERROR. No image files found in the folder.")
        return
    
    input_image_path = os.path.join(input_folder, image_files[0])
    
    # Load and preprocess the image
    processed_image = load_and_preprocess_image(input_image_path)
    
    # Save the preprocessed image
    save_image(processed_image, output_folder, output_image_name)

if __name__ == '__main__':
    process()
