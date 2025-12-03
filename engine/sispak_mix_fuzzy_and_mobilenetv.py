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
MODEL_FILENAME = 'best_coffee_mobilenet_v2.h5'
DATASET_DIR = 'dataset_ready' 

# Pastikan urutan ini SAMA dengan urutan alfabetis folder
CLASS_NAMES = ['cercospora', 'karat_daun', 'phoma', 'sehat']

# ==================== DATABASE GEJALA (SISTEM PAKAR) ====================
# Pertanyaan untuk memvalidasi setiap penyakit
DISEASE_QUESTIONS = {
    'cercospora': [
        "Apakah bercak berbentuk bulat dengan pusat berwarna abu-abu/putih?",
        "Apakah pinggiran bercak berwarna coklat kemerahan/gelap (seperti mata)?",
        "Apakah bercak juga menyerang buah kopi (membuat buah hitam/cekung)?"
    ],
    'karat_daun': [
        "Apakah terdapat bubuk/serbuk berwarna oranye atau kuning cerah?",
        "Apakah serbuk tersebut berada di bagian BAWAH permukaan daun?",
        "Apakah bercak terlihat seperti karat pada besi?"
    ],
    'phoma': [
        "Apakah bercak berwarna hitam pekat atau coklat gelap?",
        "Apakah posisi bercak berada di pinggiran/tepi daun?",
        "Apakah bercak terlihat tidak beraturan (bukan bulat sempurna)?"
    ],
    'sehat': [
        "Apakah daun terlihat hijau segar tanpa bercak?",
        "Apakah permukaan daun mulus dan tidak ada lubang/gosong?",
        "Apakah tanaman tumbuh normal tanpa tanda layu?"
    ]
}

# ==================== STEP 1: BUILD MODEL ====================
def build_mobilenet_model():
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==================== STEP 2: TRAIN MODEL ====================
def train_model():
    print("\n🚀 TRAINING MODEL...")
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'val')
    
    if not os.path.exists(train_dir):
        print(f"❌ Error: Folder {train_dir} tidak ditemukan.")
        return

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, horizontal_flip=True)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

    model = build_mobilenet_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_FILENAME, monitor='val_accuracy', save_best_only=True)
    ]
    
    model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=callbacks)
    print("✅ Training Selesai.")

# ==================== LOGIKA DIAGNOSIS HYBRID ====================
def run_symptom_check():
    """
    Mengajukan pertanyaan ke user dan menghitung skor gejala.
    Output: Dictionary skor gejala (0.0 - 1.0) untuk setiap penyakit.
    """
    print("\n🕵️ MULAI PEMERIKSAAN GEJALA FISIK")
    print("Jawab 'y' untuk Ya, atau tekan Enter untuk Tidak.")
    
    symptom_scores = {cls: 0.0 for cls in CLASS_NAMES}
    
    for disease, questions in DISEASE_QUESTIONS.items():
        print(f"\n--- Cek Gejala: {disease.upper()} ---")
        yes_count = 0
        total_q = len(questions)
        
        for q in questions:
            ans = input(f"{q} (y/n): ").lower()
            if ans == 'y':
                yes_count += 1
        
        # Hitung persentase gejala (misal 2 dari 3 = 0.66)
        symptom_scores[disease] = yes_count / total_q
    
    return symptom_scores

def predict_hybrid(image_path):
    print("\n" + "="*60)
    print("DIAGNOSIS HYBRID (GAMBAR + GEJALA)")
    print("="*60)
    
    if not os.path.exists(MODEL_FILENAME):
        print("❌ Model belum dilatih. Pilih menu 1 dulu.")
        return

    # --- 1. PREDIKSI GAMBAR (AI) ---
    try:
        model = keras.models.load_model(MODEL_FILENAME)
        img = Image.open(image_path).convert('RGB')
        img_display = img.copy()
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.expand_dims(preprocess_input(np.array(img)), axis=0)
        
        # Dapat probabilitas dari gambar (misal: [0.1, 0.8, 0.05, 0.05])
        ai_probs = model.predict(img_arr, verbose=0)[0]
    except Exception as e:
        print(f"Error baca gambar: {e}")
        return

    print("✅ Analisis Gambar Selesai.")
    
    # --- 2. PREDIKSI GEJALA (MANUSIA) ---
    user_scores = run_symptom_check()
    
    # --- 3. PENGGABUNGAN SKOR (WEIGHTED AVERAGE) ---
    # Kita beri bobot: Gambar 60%, Jawaban User 40%
    # Kamu bisa ubah bobot ini (misal 0.5 dan 0.5)
    ALPHA = 0.6 
    BETA = 0.4
    
    final_scores = {}
    print("\n--- PERHITUNGAN SKOR AKHIR ---")
    print(f"{'Penyakit':<15} | {'AI Score':<10} | {'Gejala Score':<12} | {'FINAL SCORE'}")
    print("-" * 55)

    for i, disease in enumerate(CLASS_NAMES):
        ai_score = ai_probs[i]
        human_score = user_scores[disease]
        
        # Rumus Kombinasi
        final = (ai_score * ALPHA) + (human_score * BETA)
        final_scores[disease] = final
        
        print(f"{disease:<15} | {ai_score:.2f}       | {human_score:.2f}         | {final:.2f}")

    # --- 4. HASIL FINAL ---
    best_disease = max(final_scores, key=final_scores.get)
    best_confidence = final_scores[best_disease] * 100

    print("\n" + "="*60)
    print(f"🏆 HASIL DIAGNOSIS: {best_disease.upper()}")
    print(f"📊 KEYAKINAN SISTEM: {best_confidence:.2f}%")
    print("="*60)
    
    plt.imshow(img_display)
    plt.axis('off')
    plt.title(f"Diagnosis: {best_disease} ({best_confidence:.1f}%)")
    plt.show()

# ==================== MAIN MENU ====================
if __name__ == "__main__":
    while True:
        print("\n=== SISTEM PAKAR KOPI HYBRID ===")
        print("1. Train Model AI")
        print("2. Diagnosis Lengkap (Gambar + Pertanyaan)")
        print("3. Keluar")
        
        p = input("Pilih: ")
        
        if p == '1':
            train_model()
        elif p == '2':
            img = input("Path gambar: ").replace('"', '')
            predict_hybrid(img)
        elif p == '3':
            break