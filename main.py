import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    Rescaling, RandomFlip, RandomRotation, RandomZoom,
    RandomTranslation # <<< KITA HAPUS 'RandomShearRange'
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI (SESUAIKAN DENGAN SPESIFIKASI ANDA) ---
DATASET_PATH = './dataset_padang_food/'

# Parameter dari Laporan dan Spreadsheet
IMG_HEIGHT = 227
IMG_WIDTH = 227
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # Tetap 1e-4
NUM_EPOCHS = 200       # Biarkan tinggi, EarlyStopping akan mengurusnya
TRAIN_TEST_SPLIT = 0.20
TRAIN_VAL_SPLIT = 0.20

# --- 2. PERSIAPAN DATA AUGMENTASI (LEBIH AGRESIF) ---
data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(height_factor=0.2, width_factor=0.2), # Zoom bisa + atau -
        RandomTranslation(height_factor=0.2, width_factor=0.2)
        # <<< KITA HAPUS 'RandomShearRange' DARI SINI
    ],
    name="data_augmentation_agresif",
)

# --- 3. MEMUAT DATASET ---
print("Memuat dataset...")
try:
    # 1. Split 80% Training dan 20% Testing
    train_ds = tf.keras.utils.image_dataset_from_directory(
      DATASET_PATH,
      validation_split=TRAIN_TEST_SPLIT,
      subset="training",
      seed=123,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
      DATASET_PATH,
      validation_split=TRAIN_TEST_SPLIT,
      subset="validation",
      seed=123,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Dataset dimuat. Ditemukan {num_classes} kelas: {class_names}")

    # Normalisasi pixel values
    normalization_layer = Rescaling(1./255)

    # Terapkan normalisasi
    train_ds = train_ds.map(lambda x, y: (x, y)) # Augmentasi ada di model
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Konfigurasi performa dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Pisahkan 20% data training untuk validasi
    train_size_tensor = tf.data.experimental.cardinality(train_ds)
    train_size_int = int(train_size_tensor.numpy())
    val_size_int = int(train_size_int * TRAIN_VAL_SPLIT)
    
    val_ds = train_ds.take(val_size_int)
    train_ds = train_ds.skip(val_size_int)
    
    print(f"Data training asli (795 file) dibagi menjadi:")
    print(f"   -> Set Validasi: {val_size_int * BATCH_SIZE} file (kurang lebih)")
    print(f"   -> Set Training Baru: {(train_size_int - val_size_int) * BATCH_SIZE} file (kurang lebih)")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Error saat memuat data: {e}")
    print(f"Pastikan path dataset Anda di '{DATASET_PATH}' sudah benar.")
    exit()


# --- 4. MEMBANGUN MODEL ALEXNET DENGAN L2 REGULARIZATION ---

reg = l2(0.0001) # Pertahankan L2 yang lemah

def create_alexnet(input_shape, num_classes):
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation, # Gunakan augmentasi yang agresif
        Rescaling(1./255),
        
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu', kernel_regularizer=reg),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        BatchNormalization(),

        Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=reg),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        BatchNormalization(),

        Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(4096, activation='relu', kernel_regularizer=reg),
        Dropout(0.5), # Dropout ini juga sangat penting
        Dense(4096, activation='relu', kernel_regularizer=reg),
        Dropout(0.5), # Dropout ini juga sangat penting

        Dense(num_classes, activation='softmax')
    ], name="AlexNet_L2_Augment")
    return model

print("Membangun model AlexNet...")
model = create_alexnet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=num_classes)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary() 

# --- 5. MELATIH MODEL ---
print(f"\nMemulai training selama {NUM_EPOCHS} epoch...")

start_train_time = time.time()

# Tentukan callback Early Stopping (kita jaga ini)
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=20,           # Beri kesabaran 20 epoch
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
  train_ds,
  epochs=NUM_EPOCHS,
  validation_data=val_ds,
  callbacks=[early_stopper] # Gunakan EarlyStopping
)

end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"\nTraining selesai dalam {training_time:.2f} detik.")


# --- 6. EVALUASI MODEL (MENGHITUNG SEMUA METRIK) ---
print("\nMemulai evaluasi model pada data test (20%)...")

y_true = []
y_pred_probs = []

start_test_time = time.time()

for images, labels in test_ds:
    y_true.extend(labels.numpy())
    batch_pred_probs = model.predict_on_batch(images)
    y_pred_probs.extend(batch_pred_probs)

end_test_time = time.time()
testing_time = end_test_time - start_test_time

y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

print(f"Testing (prediksi) selesai dalam {testing_time:.2f} detik.")

# --- 7. MENAMPILKAN SEMUA METRIK TUGAS ---
print("\n" + "="*30)
print("HASIL EVALUASI MODEL (UNTUK LAPORAN)")
print("="*30)

# 1. AKURASI (Tugas Arya)
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"\n1. Akurasi (Accuracy):\n   {accuracy:.4f} (atau {accuracy*100:.2f}%)")

# 2. & 3. RECALL & PRESISI (Tugas Daneeel & Dzikri)
print("\n2. & 3. Laporan Klasifikasi (Recall, Presisi, F1-Score):")
report_dict = classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0, output_dict=True)
print(classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0))

recall_macro = report_dict['macro avg']['recall']
recall_weighted = report_dict['weighted avg']['recall']
precision_macro = report_dict['macro avg']['precision']
precision_weighted = report_dict['weighted avg']['precision']
f1_macro = report_dict['macro avg']['f1-score']
f1_weighted = report_dict['weighted avg']['f1-score']

print(f"   - RECALL (Macro Avg):     {recall_macro:.4f}")
print(f"   - RECALL (Weighted Avg):  {recall_weighted:.4f}  <-- (Ini mungkin yang Anda cari)")
print(f"   - PRESISI (Macro Avg):    {precision_macro:.4f}")
print(f"   - PRESISI (Weighted Avg): {precision_weighted:.4f}  <-- (Ini mungkin yang Anda cari)")
print(f"   - F1-Score (Macro Avg):   {f1_macro:.4f}")
print(f"   - F1-Score (Weighted Avg):{f1_weighted:.4f}")

# 4. ROC/AUC (Tugas Arya)
try:
    auc_macro_ovr = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    auc_weighted_ovr = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='weighted')
    print(f"\n4. ROC/AUC (One-vs-Rest):")
    print(f"   - ROC/AUC (Macro Avg):    {auc_macro_ovr:.4f}")
    print(f"   - ROC/AUC (Weighted Avg): {auc_weighted_ovr:.4f}")
except ValueError as e:
    print(f"\n4. ROC/AUC: Tidak dapat dihitung. Error: {e}")
    print("   (Ini bisa terjadi jika ada kelas di data tes yang tidak pernah diprediksi)")

# 5. COMPUTATION TIME (Tugas George)
print(f"\n5. Waktu Komputasi (Computation Time):")
print(f"   - Waktu Training: {training_time:.2f} detik")
print(f"   - Waktu Testing:  {testing_time:.2f} detik (waktu untuk prediksi data tes)")

print("\n" + "="*30)
print("SEMUA METRIK TEREKAP")
print("="*30)
print(f"Akurasi:          {accuracy:.4f}")
print(f"Recall (Weighted):  {recall_weighted:.4f}")
print(f"Presisi (Weighted): {precision_weighted:.4f}")
print(f"F1-Score (Weighted):{f1_weighted:.4f}")
try:
    print(f"ROC/AUC (Weighted): {auc_weighted_ovr:.4f}")
except NameError:
    print("ROC/AUC (Weighted): Gagal dihitung")
print(f"Waktu Training:     {training_time:.2f} detik")
print(f"Waktu Testing:      {testing_time:.2f} detik")
print("="*30)


# --- 8. (OPSIONAL) MEMBUAT PLOT ---
print("\nMembuat plot training history dan confusion matrix...")

try:
    # Plot Akurasi & Loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('accuracy_loss_plot.png')
    print("Plot Akurasi & Loss disimpan sebagai 'accuracy_loss_plot.png'")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Plot Confusion Matrix disimpan sebagai 'confusion_matrix.png'")

except Exception as e:
    print(f"Gagal membuat plot: {e}")

print("\n--- Selesai ---")