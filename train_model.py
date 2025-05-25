import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from prepare_data import prepare_dataset

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a CNN model for e-waste detection using transfer learning
    with MobileNetV2 (efficient for CPU usage)
    """
    # Use MobileNetV2 as base model (good performance on CPU)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, X_val, y_train, y_val, epochs=15, batch_size=32):
    """
    Train the model with early stopping and checkpointing
    """
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create the model
    model = create_model()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/ewaste_detector.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model

def main():
    # Load and prepare dataset
    X_train, X_val, y_train, y_val = prepare_dataset()
    
    if X_train is None:
        print("Failed to prepare dataset. Exiting.")
        return
    
    # Train the model
    print("Training model...")
    model = train_model(X_train, X_val, y_train, y_val)
    
    # Save the model
    model.save('models/ewaste_detector.h5')
    print("Model saved to 'models/ewaste_detector.h5'")

if __name__ == "__main__":
    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    main() 