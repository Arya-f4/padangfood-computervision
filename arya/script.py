import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import time

# --- 1. Configuration ---
# Update this path to your dataset location
dataset_dir = 'dataset_padang_food' 

# Model Parameters
IMG_WIDTH = 227
IMG_HEIGHT = 227
BATCH_SIZE = 32
EPOCHS = 100 # Warning: Training from scratch needs many epochs
# --- PERBAIKAN ---
# Mengubah dari 8 menjadi 9 berdasarkan error log Anda
NUM_CLASSES = 9 # Based on your folder list (9 classes found)

# --- 2. Load and Prepare Data ---
print("Loading dataset...")
# 80% for training, 20% for validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_ds.class_names
print(f"Classes found: {class_names}")
assert len(class_names) == NUM_CLASSES, "Class count mismatch"

# Normalize data: [0, 255] -> [0, 1]
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Optimize data pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Build AlexNet (From Scratch) ---
def build_alexnet(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # L1
        layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # L2
        layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # L3
        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # L4
        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # L5
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Flatten
        layers.Flatten(),
        # FC1
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        # FC2
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_alexnet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES)

# --- 4. Compile Model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- 5. Train Model ---
print("\nStarting model training (AlexNet from scratch)...")
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS 
)

print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

# --- 6. Generate Plots (Accuracy & Loss) ---
# This part creates the first deliverable (plot)
print("Generating Accuracy/Loss plot...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save the plot to a file
plt.savefig('accuracy_loss_plot.png')
print("Saved 'accuracy_loss_plot.png'")

# --- 7. Generate Reports (Confusion Matrix & Classification Report) ---
# This part creates the other deliverables
print("Generating full validation reports...")

# Get true labels and predictions from the validation set
y_true = []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# Get model predictions
predictions_probs = model.predict(val_ds)
y_pred = np.argmax(predictions_probs, axis=1)

# Generate and print Classification Report
print("\n--- Classification Report ---")
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print(report)
print("-----------------------------")

# Generate and save Confusion Matrix plot
print("Generating Confusion Matrix plot...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Save the matrix to a file
plt.savefig('confusion_matrix.png')
print("Saved 'confusion_matrix.png'")

# Save the final model
model.save('alexnet_padang_food_scratch.h5')
print("\nModel saved as 'alexnet_padang_food_scratch.h5'")


