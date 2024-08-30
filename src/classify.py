import tensorflow as tf
import cv2
import numpy as np

def load_model(model_path="models/saved_model.h5"):
    """Load the trained model."""
    model = tf.keras.models.load_model(model_path)
    return model

def classify_image(image_path, model):
    """Classify the image using the trained model."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image / 255.0, axis=0)  # Normalize and add batch dimension
    prediction = model.predict(image)
    return np.argmax(prediction)

if __name__ == "__main__":
    model = load_model()
    image_path = "data/processed/example.jpg"
    class_id = classify_image(image_path, model)
    print(f"Classified image as class ID: {class_id}")
    print("Thank you")