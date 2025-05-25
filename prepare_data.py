import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size=(224, 224)):
    """
    Load images from the specified directory and resize them
    """
    images = []
    labels = []
    
    # Define classes: 0 = non-e-waste, 1 = e-waste
    class_mapping = {"non_ewaste": 0, "ewaste": 1}
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            label = class_mapping.get(class_name.lower(), -1)
            if label == -1:
                continue  # Skip unknown classes
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Read and resize image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, img_size)
                    img = img / 255.0  # Normalize
                    
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def prepare_dataset(dataset_dir="organized_dataset"):
    """
    Prepare dataset for training
    """
    # Dataset structure:
    # - organized_dataset/train/ewaste/
    # - organized_dataset/train/non_ewaste/
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory '{train_dir}' not found.")
        print("Please run organize_dataset.py first to prepare the dataset.")
        return None, None, None, None
    
    print("Loading training data...")
    X_train, y_train = load_data(train_dir)
    
    print("Loading validation data...")
    X_val, y_val = load_data(test_dir)
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("No images found. Please check the dataset structure.")
        return None, None, None, None
    
    # Convert labels to one-hot encoding for multi-class classification
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    
    print(f"Dataset prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = prepare_dataset()
    if X_train is not None:
        print("Data preparation complete!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}") 