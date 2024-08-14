import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape=(256, 256, 3), num_classes=3):
    """Build a simple CNN model for classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir="data/processed/", epochs=10, batch_size=32):
    """Train the CNN model."""
    datagen = ImageDataGenerator(validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    model = build_model(num_classes=train_generator.num_classes)
    model.fit(train_generator, validation_data=val_generator, epochs=epochs)
    model.save("models/saved_model.h5")
    print("Model saved to models/saved_model.h5")

if __name__ == "__main__":
    train_model()
