import os
import cv2
import numpy as np

def preprocess_image(image_path, output_path, size=(256, 256)):
    """Preprocess the image: resize and normalize."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalize to [0, 1]
    cv2.imwrite(output_path, (image * 255).astype(np.uint8))
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    input_folder = "data/raw/"
    output_folder = "data/processed/"
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        preprocess_image(os.path.join(input_folder, filename), 
                         os.path.join(output_folder, filename))
