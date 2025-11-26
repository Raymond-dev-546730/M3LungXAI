# Import required libraries
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)
    return images

def read_all_images(base_path):
    dataset = {}
    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        if os.path.isdir(folder_path):
            images = read_images_from_folder(folder_path)
            dataset[label] = images
    return dataset

# Define preprocessing pipeline
def process_images(images):
    processed_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (128, 128))  # Resize to 128x128
        img = img / 255.0  # Normalize pixel values
        processed_images.append(img)
    return processed_images

# Process images in parallel to improve performance
def process_all_images(dataset):
    processed_dataset = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_images, images): label for label, images in dataset.items()}
        for future in futures:
            label = futures[future]
            processed_dataset[label] = future.result()
    return processed_dataset

# Save processed images to a new directory
def save_images(dataset, base_path):
    for label, images in dataset.items():
        label_folder = os.path.join(base_path, label)
        os.makedirs(label_folder, exist_ok=True)
        for i, img in enumerate(images):
            save_path = os.path.join(label_folder, f'image_{i}.png')
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(save_path, img)

if __name__ == '__main__':
    # Set the path to the dataset folder
    base_path = './CT_scan'
    new_base_path = './CT_scan_processed_128x128'

    # Perform all preprocessing steps
    dataset = read_all_images(base_path)
    processed_dataset = process_all_images(dataset)

    # Save the processed images
    save_images(processed_dataset, new_base_path)
    print("Finished pre-processing and saving images.")