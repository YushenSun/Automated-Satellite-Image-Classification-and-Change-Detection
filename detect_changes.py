import cv2
import numpy as np

def detect_changes(image1_path, image2_path, output_path):
    """Detect changes between two images using image differencing."""
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    difference = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(output_path, threshold_diff)
    print(f"Change detection result saved to {output_path}")

if __name__ == "__main__":
    image1_path = "data/processed/before.jpg"
    image2_path = "data/processed/after.jpg"
    output_path = "data/results/change_map.jpg"
    detect_changes(image1_path, image2_path, output_path)
