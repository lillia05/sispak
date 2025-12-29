# ==============================================================================

# REQUIREMENTS
# DATABASE SUMBER DENGAN NAMA -> "DATASET"
# RINCIAN
# DATASET/cescospora/...
# DATASET/karat-daun/...
# DATASET/phoma/...
# DATASET/sehat/...
# DAN OUTPUNYA BERUPA FOLDER "DATASET_HASIL"

# ==============================================================================

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
import shutil
import random
from sklearn.metrics.pairwise import cosine_similarity

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 4 

# NAMA FOLDER
SOURCE_DIR = 'DATASET'        # Folder sumber (mentah)
BASE_DIR = 'DATASET_HASIL'    # Folder tujuan (akan dibuat otomatis)
MODEL_FILENAME = 'best_coffee_model.h5'
VECTOR_FILENAME = 'coffee_reference_vector.npy'

# Pastikan nama kelas sesuai dengan nama folder di DATASET
CLASS_NAMES = ['cercospora', 'karat-daun', 'phoma', 'sehat']
SIMILARITY_THRESHOLD = 0.65 

# ==================== STEP 0: AUTO SPLIT DATASET ====================
def split_dataset():
    print("\n" + "="*60)
    print("MEMPROSES & MEMBAGI DATASET (80% Train, 10% Val, 10% Test)")
    print("="*60)

    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Folder sumber '{SOURCE_DIR}' tidak ditemukan!")
        return False

    # Hapus folder hasil lama jika ada (biar fresh)
    if os.path.exists(BASE_DIR):
        print(f"🧹 Menghapus folder lama '{BASE_DIR}'...")
        shutil.rmtree(BASE_DIR)

    # Buat struktur folder baru
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(BASE_DIR, split, class_name), exist_ok=True)

    # Proses pemindahan file
    for class_name in CLASS_NAMES:
        source_class_dir = os.path.join(SOURCE_DIR, class_name)
        
        if not os.path.exists(source_class_dir):
            print(f"⚠️ Warning: Folder kelas '{class_name}' tidak ada di {SOURCE_DIR}")
            continue

        files = os.listdir(source_class_dir)
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Acak urutan file
        random.shuffle(files)
        
        # Hitung jumlah pembagian
        total = len(files)
        train_count = int(total * 0.8)
        val_count = int(total * 0.1)
        # Sisa masuk ke test
        
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        print(f"📂 Memproses '{class_name}': {total} gambar "
              f"-> (Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)})")

        # Copy file ke tujuan
        for f in train_files:
            shutil.copy(os.path.join(source_class_dir, f), os.path.join(BASE_DIR, 'train', class_name, f))
        for f in val_files:
            shutil.copy(os.path.join(source_class_dir, f), os.path.join(BASE_DIR, 'val', class_name, f))
        for f in test_files:
            shutil.copy(os.path.join(source_class_dir, f), os.path.join(BASE_DIR, 'test', class_name, f))
            
    print("\n✅ Dataset berhasil dibagi dan siap digunakan!")
    return True

# ==================== STEP 1: BUILD MODEL ====================
def build_mobilenet_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='feature_extractor')(x) 
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==================== STEP 2: TRAIN & GENERATE VECTOR ====================
def train_model():
    # 1. Split Data Dulu
    success = split_dataset()
    if not success: return

    print("\n" + "="*60)
    print("MULAI TRAINING MODEL")
    print("="*60)
    
    train_dir = os.path.join(BASE_DIR, 'train')
    val_dir = os.path.join(BASE_DIR, 'val')

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

    print("Load data dari folder hasil split...")
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
    
    history = model.fit(train_generator, epochs=EPOCHS,
                        validation_data=val_generator, callbacks=callbacks)
    
    print(f"✅ Model disimpan: {MODEL_FILENAME}")

    # GENERATE VECTOR REFERENCE (SIDIK JARI)
    print("\n🧬 Membuat 'Sidik Jari' Referensi Daun Kopi...")
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('feature_extractor').output)
    
    # Ambil sampel dari data train
    sample_images, _ = next(train_generator)
    # Jika batch size kecil, kita bisa ambil beberapa batch biar lebih akurat
    if len(sample_images) < 32: 
        print("   Mengambil sampel tambahan untuk akurasi vektor...")
        sample_images2, _ = next(train_generator)
        sample_images = np.concatenate((sample_images, sample_images2))

    features = feature_model.predict(sample_images)
    mean_feature_vector = np.mean(features, axis=0)
    
    np.save(VECTOR_FILENAME, mean_feature_vector)
    print(f"✅ Referensi Vektor disimpan: {VECTOR_FILENAME}")

# ==================== STEP 3: PREDICT (Vector Guard) ====================
def predict_image(image_path):
    print("\n" + "="*60)
    print("PREDIKSI (VECTOR GUARD + CLASSIFICATION)")
    print("="*60)
    
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(VECTOR_FILENAME):
        print("❌ Model belum siap. Jalankan Training (Menu 1) dulu.")
        return

    try:
        full_model = keras.models.load_model(MODEL_FILENAME)
        reference_vector = np.load(VECTOR_FILENAME)
        
        feature_extractor = tf.keras.Model(
            inputs=full_model.input, 
            outputs=full_model.get_layer('feature_extractor').output
        )

        img = Image.open(image_path).convert('RGB')
        img_display = img.copy()
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_preprocessed = preprocess_input(img_array.copy())
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # 1. CEK STRUKTUR (Anti Topi Hijau)
        current_features = feature_extractor.predict(img_batch, verbose=0)
        similarity_score = cosine_similarity(
            current_features.reshape(1, -1), 
            reference_vector.reshape(1, -1)
        )[0][0]

        print(f"🔍 Similarity Score: {similarity_score:.4f} (Threshold: {SIMILARITY_THRESHOLD})")

        if similarity_score < SIMILARITY_THRESHOLD:
            print("\n⛔ DITOLAK: STRUKTUR ASING")
            print("   Objek tidak dikenali sebagai daun kopi.")
            plt.imshow(img_display)
            plt.axis('off')
            plt.title(f"DITOLAK (Score: {similarity_score:.2f})", color='red')
            plt.show()
            return

        # 2. KLASIFIKASI PENYAKIT
        predictions = full_model.predict(img_batch, verbose=0)
        idx = np.argmax(predictions[0])
        confidence = predictions[0][idx] * 100
        
        # Kita ambil nama kelas dari urutan abjad (standar Keras FlowFromDirectory)
        sorted_classes = sorted(CLASS_NAMES)
        disease_name = sorted_classes[idx]

        print(f"\n✅ TERIMA: {disease_name}")
        print(f"📊 Confidence: {confidence:.2f}%")
        
        plt.imshow(img_display)
        plt.axis('off')
        plt.title(f"{disease_name} ({confidence:.1f}%)", color='green')
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ==================== MAIN ====================
if __name__ == "__main__":
    while True:
        print("\n=== SYSTEM PAKAR KOPI (AUTO SPLIT DATASET) ===")
        print("1. Split Dataset & Train Model (Akan membuat folder DATASET_HASIL)")
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