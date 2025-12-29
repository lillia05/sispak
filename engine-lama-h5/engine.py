import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4
MODEL_FILENAME = 'best_coffee_mobilenet_v2.h5' # Saya ganti nama file biar fresh
DATASET_DIR = 'dataset_ready' 

CLASS_NAMES = [
    'cercospora', 
    'karat_daun', 
    'phoma', 
    'sehat'
]

# ==================== STEP 1: BUILD MODEL (MobileNetV2) ====================
def build_mobilenet_model():
    # 1. Load Base Model
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, 
        weights='imagenet'
    )
    
    base_model.trainable = False 

    # 2. Arsitektur Head (Tanpa layer preprocessing di dalam)
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Langsung masuk ke base model (preprocessing dilakukan di luar)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================== STEP 2: TRAIN MODEL ====================
def train_model():
    print("\n" + "="*60)
    print("TRAINING DENGAN MOBILENET V2 (FIXED)")
    print("="*60)
    
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'val')

    if not os.path.exists(train_dir):
        print(f"❌ Error: Folder '{train_dir}' tidak ditemukan.")
        return

    # PERUBAHAN PENTING: preprocessing_function ditaruh di sini!
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # <-- INI KUNCINYA
        rotation_range=20,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input # <-- Validasi juga wajib pakai ini
    ) 

    print("Load data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical'
    )

    model = build_mobilenet_model()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_FILENAME, monitor='val_accuracy', save_best_only=True)
    ]
    
    print("\n🚀 Mulai Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Plot Grafik
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Result')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"✅ Model disimpan sebagai: {MODEL_FILENAME}")

# ==================== STEP 3: PREDICT ====================
def predict_image(image_path):
    print("\n" + "="*60)
    print("PREDIKSI (MOBILENET V2)")
    print("="*60)
    
    if not os.path.exists(MODEL_FILENAME):
        print("❌ Model tidak ditemukan. WAJIB Training dulu (Pilih menu 1)!")
        return

    try:
        # Load model sekarang aman karena tidak ada layer aneh di dalamnya
        model = keras.models.load_model(MODEL_FILENAME)
        
        img = Image.open(image_path).convert('RGB')
        img_display = img.copy()
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        img_array = np.array(img)
        
        # PERUBAHAN PENTING: Preprocessing manual sebelum masuk model
        # Ini mengubah pixel dari 0-255 menjadi -1 s/d 1 (standar MobileNet)
        img_array = preprocess_input(img_array) 
        
        img_array = np.expand_dims(img_array, axis=0) 

        predictions = model.predict(img_array, verbose=0)
        idx = np.argmax(predictions[0])
        confidence = predictions[0][idx] * 100
        
        sorted_classes = sorted(CLASS_NAMES)
        disease_name = sorted_classes[idx]

        print(f"\n🏷️ HASIL: {disease_name}")
        print(f"📊 CONFIDENCE: {confidence:.2f}%")
        
        plt.imshow(img_display)
        plt.axis('off')
        plt.title(f"{disease_name} ({confidence:.1f}%)")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        print("Saran: Coba training ulang (Menu 1) untuk memperbarui file model.")

# ==================== MAIN ====================
if __name__ == "__main__":
    while True:
        print("\n=== SYSTEM PAKAR KOPI (MobileNetV2 FIXED) ===")
        print("1. Train Model (WAJIB DILAKUKAN ULANG)")
        print("2. Prediksi Gambar")
        print("3. Keluar")
        
        p = input("Pilih: ")
        
        if p == '1':
            train_model()
        elif p == '2':
            img = input("Path gambar: ").replace('"', '')
            predict_image(img)
        elif p == '3':
            break