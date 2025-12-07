from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max 5MB

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
MODEL_FILENAME = 'best_coffee_mobilenet_v2.h5'
CLASS_NAMES = ['cercospora', 'karat_daun', 'phoma', 'sehat']

# Database Penyakit dengan Nama Indonesia
DISEASE_INFO = {
    'cercospora': {
        'name': 'Cercospora (Bercak Daun)',
        'description': 'Penyakit jamur yang menyebabkan bercak bulat dengan pusat abu-abu dan tepi coklat kemerahan.',
        'treatment': 'Gunakan fungisida berbahan copper, pangkas daun terinfeksi, dan jaga sanitasi kebun.'
    },
    'karat_daun': {
        'name': 'Karat Daun Kopi (Coffee Leaf Rust)',
        'description': 'Penyakit jamur yang ditandai bubuk oranye/kuning di bawah permukaan daun.',
        'treatment': 'Aplikasi fungisida sistemik, tingkatkan sirkulasi udara, dan gunakan varietas tahan penyakit.'
    },
    'phoma': {
        'name': 'Phoma (Bercak Hitam)',
        'description': 'Infeksi jamur yang menimbulkan bercak hitam tidak beraturan di tepi daun.',
        'treatment': 'Potong bagian terinfeksi, gunakan fungisida, hindari kelembaban berlebih.'
    },
    'sehat': {
        'name': 'Tanaman Sehat',
        'description': 'Tanaman kopi dalam kondisi sehat tanpa tanda-tanda penyakit.',
        'treatment': 'Lanjutkan perawatan rutin: pemupukan, penyiraman, dan monitoring berkala.'
    }
}

# Bobot untuk hybrid diagnosis (Analisis vs Manual)
ALPHA = 0.6  # Bobot Analisis
BETA = 0.4   # Bobot Gejala Manual

# ==================== LOAD MODEL ====================
model = None
def load_model():
    global model
    if model is None and os.path.exists(MODEL_FILENAME):
        model = keras.models.load_model(MODEL_FILENAME)
        print("✅ Model berhasil dimuat")
    elif not os.path.exists(MODEL_FILENAME):
        print("⚠️ Model tidak ditemukan. Jalankan training terlebih dahulu.")

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', active_page='index')

@app.route('/diagnosa')
def diagnosa():
    return render_template('diagnosa.html', active_page='diagnosa')

@app.route('/penyakit')
def penyakit():
    return render_template('penyakit.html', active_page='penyakit')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html', active_page='tentang')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model belum dimuat. Pastikan file best_coffee_mobilenet_v2.h5 tersedia.'}), 500

        # 1. AMBIL GAMBAR
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada gambar yang diupload'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'File kosong'}), 400

        # Baca dan proses gambar
        img = Image.open(file.stream).convert('RGB')
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.expand_dims(preprocess_input(np.array(img_resized)), axis=0)
        
        # Prediksi Analisis
        ai_probs = model.predict(img_arr, verbose=0)[0]

        # 2. AMBIL NILAI GEJALA MANUAL
        gejala_values = []
        for i in range(1, 9):  # Loop 1 sampai 8
            val = request.form.get(f'gejala_{i}', '0')
            gejala_values.append(float(val))

        # Mapping gejala ke penyakit (logika sederhana)
        manual_scores = calculate_manual_scores(gejala_values)

        # 3. GABUNGKAN SKOR (HYBRID)
        final_scores = {}
        for i, disease in enumerate(CLASS_NAMES):
            ai_score = float(ai_probs[i])
            manual_score = manual_scores.get(disease, 0.0)
            final_score = (ai_score * ALPHA) + (manual_score * BETA)
            final_scores[disease] = final_score

        # 4. TENTUKAN HASIL
        best_disease = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_disease] * 100

        # 5. SIAPKAN RESPONSE
        result = {
            'diagnosis': best_disease,
            'confidence': round(confidence, 2),
            'disease_name': DISEASE_INFO[best_disease]['name'],
            'description': DISEASE_INFO[best_disease]['description'],
            'treatment': DISEASE_INFO[best_disease]['treatment'],
            'ai_scores': {CLASS_NAMES[i]: round(float(ai_probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
            'manual_scores': {k: round(v * 100, 2) for k, v in manual_scores.items()},
            'final_scores': {k: round(v * 100, 2) for k, v in final_scores.items()}
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== LOGIKA FUZZY MANUAL ====================
def calculate_manual_scores(gejala_values):
    """
    Mapping Gejala (0-1):
    0: Tekstur Bubuk (Rust)       4: Posisi Pinggir (Phoma)
    1: Warna Oranye (Rust)        5: Warna Hitam (Phoma)
    2: Mata Ayam (Cercospora)     6: Halo Kuning (Cercospora)
    3: Bentuk Bulat (Cercospora)  7: Luas Kerusakan (Sehat Check)
    """
    scores = {cls: 0.0 for cls in CLASS_NAMES}
    
    # --- 1. KARAT DAUN (Leaf Rust) ---
    # Ciri Mutlak: Bubuk (0) & Warna Oranye (1)
    scores['karat_daun'] += gejala_values[0] * 1.0  # Bobot Tertinggi (Signature)
    scores['karat_daun'] += gejala_values[1] * 0.8  # Warna Oranye
    
    # --- 2. CERCOSPORA (Brown Eye Spot) ---
    # Ciri Mutlak: Mata Ayam (2), Bulat (3), Halo Kuning (6)
    scores['cercospora'] += gejala_values[2] * 0.9  # Bobot Tertinggi (Signature)
    scores['cercospora'] += gejala_values[3] * 0.6  # Bentuk Bulat
    scores['cercospora'] += gejala_values[6] * 0.4  # Halo Kuning (Sering muncul)
    
    # --- 3. PHOMA (Leaf Blight) ---
    # Ciri Mutlak: Pinggir Daun (4), Warna Hitam (5), Bentuk Tak Beraturan
    scores['phoma'] += gejala_values[4] * 0.9       # Bobot Tertinggi (Signature)
    scores['phoma'] += gejala_values[5] * 0.7       # Warna Hitam
    
    # Logika Terbalik: Jika bentuk TIDAK bulat (nilai rendah), skor Phoma naik
    if gejala_values[3] < 0.4: 
        scores['phoma'] += 0.4

    # --- 4. SEHAT (Healthy) ---
    # Logika: Kerusakan (7) sangat rendah DAN tidak ada gejala spesifik (0, 2, 4)
    severity = gejala_values[7]
    key_symptoms = max(gejala_values[0], gejala_values[2], gejala_values[4])
    
    if severity < 0.15 and key_symptoms < 0.2:
        scores['sehat'] = 0.95
    elif severity < 0.3:
        scores['sehat'] = 0.6
    else:
        scores['sehat'] = 0.1

    # --- NORMALISASI ---
    max_val = max(scores.values()) if max(scores.values()) > 0 else 1.0
    for k in scores:
        scores[k] = round(scores[k] / max_val, 2)
        
    return scores
# ==================== JALANKAN APP ====================
if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)