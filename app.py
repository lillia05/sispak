from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os
import cv2  # Library untuk pengolahan citra (Validasi Warna)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max 5MB

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
MODEL_FILENAME = 'best_coffee_mobilenet_v2-new.h5'
CLASS_NAMES = ['cercospora', 'karat_daun', 'phoma', 'sehat']

# Database Penyakit & Solusi
DISEASE_INFO = {
    'cercospora': {
        'name': 'Cercospora (Mata Ayam)',
        'description': 'Bercak daun dengan pusat berwarna abu-abu/putih (mata) dikelilingi halo kuning. Sering terjadi pada pembibitan atau tanaman kurang naungan.',
        'treatment': 'Pangkas daun sakit, atur naungan agar tidak terlalu lembab, gunakan fungisida bahan aktif Tembaga atau Azoksistrobin.'
    },
    'karat_daun': {
        'name': 'Karat Daun (Leaf Rust)',
        'description': 'Terdapat serbuk/tepung berwarna oranye jingga di permukaan bawah daun. Daun menguning dan gugur lebih awal.',
        'treatment': 'Gunakan varietas tahan karat (misal: Sigarar Utang), pemupukan seimbang, dan fungisida sistemik (Triazol) jika serangan parah.'
    },
    'phoma': {
        'name': 'Phoma (American Leaf Spot)',
        'description': 'Bercak hitam pekat tidak beraturan, seringkali membuat pinggiran daun keriting/gosong/kering. Bisa menyerang pucuk muda.',
        'treatment': 'Kurangi kelembaban kebun, pangkas tunas air, semprot fungisida saat pembentukan buah.'
    },
    'sehat': {
        'name': 'Tanaman Sehat',
        'description': 'Daun hijau segar, permukaan rata, tidak ada bercak, serbuk, atau kerusakan fisik yang signifikan.',
        'treatment': 'Pertahankan perawatan rutin (pemupukan berimbang & sanitasi) dan lakukan monitoring berkala.'
    }
}

# Bobot Hybrid (30% AI, 70% Rules)
ALPHA = 0.3  # Bobot AI
BETA = 0.7   # Bobot Sistem Pakar

# ==================== LOAD MODEL ====================
model = None
def load_model():
    global model
    if model is None and os.path.exists(MODEL_FILENAME):
        model = keras.models.load_model(MODEL_FILENAME)
        print("✅ Model AI berhasil dimuat")
    elif not os.path.exists(MODEL_FILENAME):
        print("⚠️ Model tidak ditemukan. Pastikan file .h5 ada di folder project.")

# ==================== VALIDASI GAMBAR (HSV) ====================
def validate_leaf_color(img_pil):
    """
    Validasi apakah gambar memiliki unsur warna daun (Hijau/Kuning/Coklat).
    Mencegah user upload foto selfie, benda mati, atau hewan.
    """
    try:
        # Konversi PIL ke OpenCV format
        img_arr = np.array(img_pil)
        # Convert RGB to HSV
        hsv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)

        # Rentang warna daun (Kuning Layu s/d Hijau Tua)
        # H: 20-90, S: 30-255, V: 30-255
        lower_bound = np.array([20, 30, 30])
        upper_bound = np.array([90, 255, 255])

        # Masking pixel yang sesuai
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        
        # Hitung rasio pixel daun vs total pixel
        leaf_ratio = np.sum(mask > 0) / mask.size

        print(f"[Validasi] Rasio Warna Daun: {leaf_ratio:.2%}")

        # Ambang batas 2% (0.02) - Sangat toleran untuk foto close-up makro
        if leaf_ratio < 0.02:
            return False, f"Gambar tidak valid (Deteksi Daun: {leaf_ratio:.1%}). Harap upload foto daun kopi yang jelas."
        
        return True, None
    except Exception as e:
        print(f"Error validasi: {e}")
        return True, None # Loloskan jika validasi error (fallback)

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
            return jsonify({'error': 'Model AI belum dimuat'}), 500

        # --- 1. PROSES GAMBAR (AI) ---
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada gambar'}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        # [VALIDASI] Cek apakah ini gambar daun
        is_valid, err_msg = validate_leaf_color(img)
        if not is_valid:
            return jsonify({'error': err_msg}), 400

        # Preprocessing untuk MobileNetV2
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.expand_dims(preprocess_input(np.array(img_resized)), axis=0)
        
        # Prediksi AI
        ai_probs = model.predict(img_arr, verbose=0)[0]

        # --- 2. PROSES GEJALA (SISTEM PAKAR FUZZY) ---
        inputs = {
            'tekstur_bubuk': float(request.form.get('tekstur_bubuk', 0)),
            'warna_oranye': float(request.form.get('warna_oranye', 0)),
            'pusat_putih': float(request.form.get('pusat_putih', 0)),
            'halo_kuning': float(request.form.get('halo_kuning', 0)),
            'bercak_hitam': float(request.form.get('bercak_hitam', 0)),
            'pinggiran_keriting': float(request.form.get('pinggiran_keriting', 0)),
            'posisi_pucuk': float(request.form.get('posisi_pucuk', 0))
        }

        manual_scores = calculate_expert_rules_fuzzy(inputs)

        # --- 3. HITUNG SKOR AKHIR (HYBRID) ---
        final_scores = {}
        for i, disease in enumerate(CLASS_NAMES):
            ai_score = float(ai_probs[i])
            manual_score = manual_scores.get(disease, 0.0)
            
            # Rumus Hybrid: (AI * 0.3) + (Pakar * 0.7)
            final_score = (ai_score * ALPHA) + (manual_score * BETA)
            final_scores[disease] = final_score

        # Tentukan Pemenang
        best_disease = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_disease] * 100

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
        print(e)
        return jsonify({'error': str(e)}), 500

# ==================== FUZZY LOGIC ENGINE ====================

def trapmf(x, a, b, c, d):
    """Fungsi Keanggotaan Trapesium"""
    if x <= a or x >= d: return 0.0
    elif a < x < b: return (x - a) / (b - a)
    elif b <= x <= c: return 1.0
    elif c < x < d: return (d - x) / (d - c)
    return 0.0

def trimf(x, a, b, c):
    """Fungsi Keanggotaan Segitiga"""
    if x <= a or x >= c: return 0.0
    elif a < x <= b: return (x - a) / (b - a)
    elif b < x < c: return (c - x) / (c - b)
    return 0.0

def calculate_expert_rules_fuzzy(i):
    """
    Sistem Pakar Fuzzy Logic (REVISED & BALANCED)
    Menghindari bias Karat/Phoma dengan logika strict trigger.
    """
    scores = {'karat_daun': 0.0, 'cercospora': 0.0, 'phoma': 0.0, 'sehat': 0.0}

    # 1. FUZZIFICATION (Ubah input 0-1 jadi derajat Rendah/Sedang/Tinggi)
    fuzzy_vars = {}
    for key, val in i.items():
        fuzzy_vars[key] = {
            'rendah': trapmf(val, -0.1, 0.0, 0.2, 0.45),
            'sedang': trimf(val, 0.25, 0.5, 0.75),
            'tinggi': trapmf(val, 0.55, 0.8, 1.0, 1.1)
        }

    def get_f(var, set_name): return fuzzy_vars[var][set_name]
    def fuzzy_or(*args): return max(args)
    def fuzzy_and(*args): return min(args)

    # =================================================================
    # RULE EVALUATION
    # =================================================================

    # --- 1. KARAT DAUN (Trigger: Bubuk & Oranye) ---
    r01 = fuzzy_and(get_f('tekstur_bubuk', 'tinggi'), get_f('warna_oranye', 'tinggi'))
    r02 = fuzzy_and(
        fuzzy_or(get_f('tekstur_bubuk', 'tinggi'), get_f('warna_oranye', 'tinggi')),
        fuzzy_or(get_f('tekstur_bubuk', 'sedang'), get_f('warna_oranye', 'sedang'))
    )
    # Strict Rule: Hanya oranye tinggi, tanpa hitam
    r03 = fuzzy_and(get_f('warna_oranye', 'tinggi'), get_f('bercak_hitam', 'rendah')) * 0.7
    
    scores['karat_daun'] = fuzzy_or(r01, r02, r03)

    # --- 2. CERCOSPORA (Trigger: Pusat Putih & Halo) ---
    r05 = fuzzy_and(get_f('pusat_putih', 'tinggi'), get_f('halo_kuning', 'tinggi'))
    r06 = fuzzy_and(get_f('pusat_putih', 'tinggi'), get_f('halo_kuning', 'sedang'))
    r07 = fuzzy_and(get_f('pusat_putih', 'tinggi'), get_f('pinggiran_keriting', 'rendah'))
    r14 = fuzzy_and(get_f('halo_kuning', 'tinggi'), get_f('tekstur_bubuk', 'rendah'))
    
    scores['cercospora'] = fuzzy_or(r05, r06, r07, r14)

    # --- 3. PHOMA (Trigger: Hitam & Keriting) ---
    r09 = fuzzy_and(get_f('bercak_hitam', 'tinggi'), get_f('pinggiran_keriting', 'tinggi'))
    r10 = fuzzy_and(get_f('posisi_pucuk', 'tinggi'), get_f('pinggiran_keriting', 'tinggi'))
    r11 = fuzzy_and(get_f('bercak_hitam', 'tinggi'), get_f('posisi_pucuk', 'tinggi'))
    r12 = fuzzy_and(get_f('bercak_hitam', 'sedang'), get_f('pinggiran_keriting', 'sedang'))
    # Pembeda: Hitam tanpa Putih
    r19 = fuzzy_and(get_f('bercak_hitam', 'tinggi'), get_f('pusat_putih', 'rendah'))
    
    scores['phoma'] = fuzzy_or(r09, r10, r11, r12, r19)

    # --- 4. KONDISI SEHAT ---
    # Logika Terbalik: Sehat = 1.0 dikurangi Gejala Tertinggi yang muncul
    gejala_maksimal = fuzzy_or(
        get_f('tekstur_bubuk', 'sedang'), get_f('tekstur_bubuk', 'tinggi'),
        get_f('warna_oranye', 'sedang'), get_f('warna_oranye', 'tinggi'),
        get_f('pusat_putih', 'sedang'), get_f('pusat_putih', 'tinggi'),
        get_f('bercak_hitam', 'sedang'), get_f('bercak_hitam', 'tinggi'),
        get_f('pinggiran_keriting', 'sedang'), get_f('pinggiran_keriting', 'tinggi')
    )
    
    scores['sehat'] = max(0, 1.0 - gejala_maksimal)

    return scores

if __name__ == '__main__':
    load_model()
    # Gunakan 0.0.0.0 agar bisa diakses di jaringan lokal jika perlu
    app.run(debug=True, host='0.0.0.0', port=5000)