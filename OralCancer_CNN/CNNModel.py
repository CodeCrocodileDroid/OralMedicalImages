import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os

# Paths
data_path = "C:/Users/mesho/PyCharmMiscProject/MedicalAI/OralCancer_Efficient/dataset/test"
save_model_path = "C:/Users/mesho/PyCharmMiscProject/MedicalAI/OralCancer_CNN/cnn_scratch_model.h5"
save_weights_path = "C:/Users/mesho/PyCharmMiscProject/MedicalAI/OralCancer_CNN/cnn_scratch_weights.weights.h5"  # Fixed extension

# Parameters
img_size = (224, 224)
batch_size = 16
epochs = 100
validation_split = 0.2

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=validation_split,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=validation_split
)

# Data generators
train_gen = train_datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    class_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    shuffle=True,
    subset="training"
)

val_gen = val_datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    class_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    shuffle=False,
    subset="validation"
)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(zip(np.unique(train_gen.classes), class_weights))
print(f"Class weights: {class_weights_dict}")
print(f"Number of classes: {train_gen.num_classes}")


# Build CNN from scratch
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fifth Convolutional Block
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model


# Create model
model = create_cnn_model((img_size[0], img_size[1], 3), train_gen.num_classes)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Display model architecture
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
    ModelCheckpoint(save_weights_path, monitor='val_accuracy', save_best_only=True,
                    save_weights_only=True, mode='max', verbose=1)
]

# Train the model
print("Training CNN from scratch...")
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

# Save the final model
model.save(save_model_path)
print(f"Training complete! Model saved to {save_model_path}")

# Plot training history
plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot AUC
plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig('cnn_training_history.png')
plt.show()

# Final evaluation
print("\n=== FINAL EVALUATION ===")
val_results = model.evaluate(val_gen, verbose=0)
print(f"Validation Loss: {val_results[0]:.4f}")
print(f"Validation Accuracy: {val_results[1]:.4f}")
print(f"Validation Precision: {val_results[2]:.4f}")
print(f"Validation Recall: {val_results[3]:.4f}")
print(f"Validation AUC: {val_results[4]:.4f}")

# Make some predictions to see results
print("\n=== SAMPLE PREDICTIONS ===")
val_images, val_labels = next(val_gen)
predictions = model.predict(val_images, verbose=0)

for i in range(min(5, len(val_images))):
    true_class = np.argmax(val_labels[i])
    pred_class = np.argmax(predictions[i])
    confidence = np.max(predictions[i])
    print(f"Sample {i + 1}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.3f}")

print("\nTraining completed successfully! 🎉")