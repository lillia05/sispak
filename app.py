from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity # <-- WAJIB ADA (Install: pip install scikit-learn)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max 5MB

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
# Gunakan nama file dari hasil training terakhir
MODEL_FILENAME = 'best_coffee_model.h5' 
VECTOR_FILENAME = 'coffee_reference_vector.npy' # <-- File sidik jari referensi

CLASS_NAMES = ['cercospora', 'karat_daun', 'phoma', 'sehat'] # Sesuaikan urutan folder

# Ambang batas kemiripan struktur (Harus > 0.65 agar dianggap daun kopi)
SIMILARITY_THRESHOLD = 0.65 

# Database Penyakit & Solusi
DISEASE_INFO = {
    'cercospora': {
        'name': 'Cercospora (Mata Ayam)',
        'description': 'Bercak daun dengan pusat berwarna abu-abu/putih (mata) dikelilingi halo kuning.',
        'treatment': 'Pangkas daun sakit, atur naungan, gunakan fungisida Tembaga/Azoksistrobin.'
    },
    'karat-daun': { # Perhatikan nama key harus sama dengan folder (pake strip atau underscore sesuaikan)
        'name': 'Karat Daun (Leaf Rust)',
        'description': 'Terdapat serbuk/tepung oranye di bawah daun. Daun menguning dan gugur.',
        'treatment': 'Gunakan varietas tahan karat, pemupukan seimbang, fungisida Triazol.'
    },
    'phoma': {
        'name': 'Phoma (American Leaf Spot)',
        'description': 'Bercak hitam pekat tidak beraturan, pinggiran daun keriting/gosong.',
        'treatment': 'Kurangi kelembaban, pangkas tunas air, fungisida saat pembentukan buah.'
    },
    'sehat': {
        'name': 'Tanaman Sehat',
        'description': 'Daun hijau segar, permukaan rata, tidak ada bercak signifikan.',
        'treatment': 'Pertahankan perawatan rutin (pemupukan & sanitasi).'
    }
}
# Mapping nama folder ke key dictionary (jika ada beda _ atau -)
CLASS_MAPPING = {
    'cercospora': 'cercospora',
    'karat-daun': 'karat-daun', # Sesuaikan dengan nama folder di dataset
    'phoma': 'phoma',
    'sehat': 'sehat'
}

# Bobot Hybrid
ALPHA = 0.3  # Bobot AI
BETA = 0.7   # Bobot Sistem Pakar

# ==================== LOAD MODEL & VECTOR ====================
model = None
feature_extractor = None
reference_vector = None

def load_resources():
    global model, feature_extractor, reference_vector
    
    # 1. Load Model Utama
    if os.path.exists(MODEL_FILENAME):
        model = keras.models.load_model(MODEL_FILENAME)
        
        # 2. Buat Feature Extractor (Ambil layer sebelum klasifikasi)
        # Pastikan saat training layer pooling diberi nama 'feature_extractor'
        try:
            feature_extractor = tf.keras.Model(
                inputs=model.input, 
                outputs=model.get_layer('feature_extractor').output
            )
            print("✅ Model AI & Feature Extractor dimuat.")
        except ValueError:
            print("⚠️ Warning: Layer 'feature_extractor' tidak ditemukan. Cek kode training.")

    else:
        print(f"❌ Error: {MODEL_FILENAME} tidak ditemukan!")

    # 3. Load Vector Referensi
    if os.path.exists(VECTOR_FILENAME):
        reference_vector = np.load(VECTOR_FILENAME)
        print("✅ Vector Referensi (Sidik Jari) dimuat.")
    else:
        print(f"❌ Error: {VECTOR_FILENAME} tidak ditemukan! Lakukan training ulang.")

# Panggil fungsi load saat aplikasi start
load_resources()

# ==================== VALIDASI 1: WARNA (HSV) ====================
def validate_leaf_color(img_pil):
    """Filter kasar untuk membuang gambar yang jelas-jelas bukan daun (misal foto selfie/tembok)"""
    try:
        img_arr = np.array(img_pil)
        hsv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        
        # Rentang hijau/kuning daun
        lower_bound = np.array([20, 30, 30])
        upper_bound = np.array([90, 255, 255])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        leaf_ratio = np.sum(mask > 0) / mask.size

        if leaf_ratio < 0.02: # Toleransi 2%
            return False, f"Warna dominan tidak sesuai daun (Rasio: {leaf_ratio:.1%})."
        return True, None
    except:
        return True, None 

# ==================== ROUTES ====================
@app.route('/')
def index(): return render_template('index.html', active_page='index')

@app.route('/diagnosa')
def diagnosa(): return render_template('diagnosa.html', active_page='diagnosa')

@app.route('/penyakit')
def penyakit(): return render_template('penyakit.html', active_page='penyakit')

@app.route('/tentang')
def tentang(): return render_template('tentang.html', active_page='tentang')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or reference_vector is None:
            return jsonify({'error': 'Model/Vector belum siap. Hubungi admin.'}), 500

        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada gambar'}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        # --- TAHAP 1: VALIDASI WARNA (Cepat) ---
        is_valid_color, err_msg = validate_leaf_color(img)
        if not is_valid_color:
            return jsonify({'error': f"Ditolak Filter Warna: {err_msg}"}), 400

        # Preprocessing
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img_resized)
        img_preprocessed = preprocess_input(img_arr.copy())
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # --- TAHAP 2: VALIDASI STRUKTUR / VECTOR GUARD (Lambat tapi Akurat) ---
        # Ekstrak fitur gambar user
        current_features = feature_extractor.predict(img_batch, verbose=0)
        
        # Hitung kemiripan dengan referensi daun kopi
        similarity_score = cosine_similarity(
            current_features.reshape(1, -1), 
            reference_vector.reshape(1, -1)
        )[0][0]

        print(f"🔍 Similarity Score: {similarity_score:.4f}")

        if similarity_score < SIMILARITY_THRESHOLD:
            return jsonify({
                'error': f"Objek tidak dikenali sebagai Daun Kopi. (Kemiripan Struktur: {similarity_score:.2f}, Min: {SIMILARITY_THRESHOLD}). Silakan upload foto daun kopi yang jelas."
            }), 400

        # --- TAHAP 3: PREDIKSI PENYAKIT (AI) ---
        ai_probs = model.predict(img_batch, verbose=0)[0]

        # --- TAHAP 4: SISTEM PAKAR FUZZY ---
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

        # --- TAHAP 5: HYBRID CALCULATION ---
        # --- TAHAP 5: HYBRID CALCULATION (FIXED NaN) ---
        final_scores = {}
        
        # Mapping untuk menangani beda nama folder vs variabel
        # Kiri: Nama folder (dari AI), Kanan: Nama key di Fuzzy Logic (Variabel)
        KEY_NORMALIZER = {
            'karat-daun': 'karat_daun',  # Ini yang sering bikin error/NaN
            'karat_daun': 'karat_daun',  # Jaga-jaga kalau namanya sudah underscore
            'cercospora': 'cercospora',
            'phoma': 'phoma',
            'sehat': 'sehat'
        }

        for i, raw_class_name in enumerate(CLASS_NAMES):
            # 1. Ambil Skor AI (Pastikan tidak NaN)
            score_ai = float(ai_probs[i])
            if np.isnan(score_ai): score_ai = 0.0
            
            # 2. Ambil Skor Manual (Sistem Pakar)
            # Kita normalkan dulu kuncinya agar cocok dengan output fuzzy logic
            fuzzy_key = KEY_NORMALIZER.get(raw_class_name, raw_class_name)
            score_manual = float(manual_scores.get(fuzzy_key, 0.0))
            if np.isnan(score_manual): score_manual = 0.0

            # 3. Hitung Hybrid
            final_score = (score_ai * ALPHA) + (score_manual * BETA)
            
            # Simpan dengan kunci yang dinormalisasi (pakai underscore biar aman di JS)
            final_scores[fuzzy_key] = final_score

        # Tentukan Pemenang
        best_disease_key = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_disease_key] * 100
        
        # Ambil Info Penyakit (Pastikan DISEASE_INFO punya key yang cocok)
        # Kita pakai get() dengan fallback ke 'sehat' biar tidak error 500
        info = DISEASE_INFO.get(best_disease_key)
        
        # Jika info masih None (misal karena typo di DISEASE_INFO), coba cari variasi key
        if info is None and best_disease_key == 'karat_daun':
             info = DISEASE_INFO.get('karat-daun') # Coba cari versi strip
        
        if info is None: # Fallback terakhir
             info = DISEASE_INFO['sehat']

        result = {
            'diagnosis': best_disease_key,
            'confidence': round(confidence, 2),
            'disease_name': info['name'],
            'description': info['description'],
            'treatment': info['treatment'],
            'similarity_info': f"Lolos Validasi Struktur ({similarity_score:.2f})",
            # Debugging scores (Opsional, biar tau nilai aslinya)
            'ai_scores': {k: round(float(v)*100, 2) for k, v in zip(CLASS_NAMES, ai_probs)},
            'manual_scores': {k: round(v * 100, 2) for k, v in manual_scores.items()},
            'final_scores': {k: round(v * 100, 2) for k, v in final_scores.items()}
        }

        return jsonify(result)

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# ==================== FUZZY LOGIC ENGINE (SAMA SEPERTI SEBELUMNYA) ====================
def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    elif a < x < b: return (x - a) / (b - a)
    elif b <= x <= c: return 1.0
    elif c < x < d: return (d - x) / (d - c)
    return 0.0

def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    elif a < x <= b: return (x - a) / (b - a)
    elif b < x < c: return (c - x) / (c - b)
    return 0.0

def calculate_expert_rules_fuzzy(i):
    # Pastikan inisialisasi menggunakan UNDERSCORE (_), jangan strip (-)
    scores = {
        'karat_daun': 0.0, 
        'cercospora': 0.0, 
        'phoma': 0.0, 
        'sehat': 0.0
    }

    # Fungsi keanggotaan (Tetap sama)
    fuzzy_vars = {}
    for key, val in i.items():
        # Validasi input: Jika input NaN, anggap 0
        safe_val = 0.0 if np.isnan(val) else val
        
        fuzzy_vars[key] = {
            'rendah': trapmf(safe_val, -0.1, 0.0, 0.2, 0.45),
            'sedang': trimf(safe_val, 0.25, 0.5, 0.75),
            'tinggi': trapmf(safe_val, 0.55, 0.8, 1.0, 1.1)
        }

    def get_f(var, set_name): return fuzzy_vars.get(var, {}).get(set_name, 0.0)
    def fuzzy_or(*args): return max(args) if args else 0.0
    def fuzzy_and(*args): return min(args) if args else 0.0

    # RULE 1: KARAT DAUN (Gunakan key: karat_daun)
    r01 = fuzzy_and(get_f('tekstur_bubuk', 'tinggi'), get_f('warna_oranye', 'tinggi'))
    r02 = fuzzy_and(
        fuzzy_or(get_f('tekstur_bubuk', 'tinggi'), get_f('warna_oranye', 'tinggi')),
        fuzzy_or(get_f('tekstur_bubuk', 'sedang'), get_f('warna_oranye', 'sedang'))
    )
    # Tambahan pengaman rule
    scores['karat_daun'] = fuzzy_or(r01, r02)

    # RULE 2: CERCOSPORA
    r05 = fuzzy_and(get_f('pusat_putih', 'tinggi'), get_f('halo_kuning', 'tinggi'))
    r06 = fuzzy_and(get_f('pusat_putih', 'tinggi'), get_f('halo_kuning', 'sedang'))
    scores['cercospora'] = fuzzy_or(r05, r06)

    # RULE 3: PHOMA
    r09 = fuzzy_and(get_f('bercak_hitam', 'tinggi'), get_f('pinggiran_keriting', 'tinggi'))
    r10 = fuzzy_and(get_f('posisi_pucuk', 'tinggi'), get_f('pinggiran_keriting', 'tinggi'))
    scores['phoma'] = fuzzy_or(r09, r10)

    # RULE 4: SEHAT
    gejala_maksimal = fuzzy_or(
        get_f('tekstur_bubuk', 'sedang'), get_f('tekstur_bubuk', 'tinggi'),
        get_f('warna_oranye', 'sedang'), get_f('warna_oranye', 'tinggi'),
        get_f('pusat_putih', 'sedang'), get_f('pusat_putih', 'tinggi'),
        get_f('bercak_hitam', 'sedang'), get_f('bercak_hitam', 'tinggi'),
        get_f('pinggiran_keriting', 'sedang'), get_f('pinggiran_keriting', 'tinggi')
    )
    scores['sehat'] = max(0.0, 1.0 - gejala_maksimal)

    return scores

if __name__ == '__main__':
    # Pastikan file model & vector ada di folder yang sama
    if not os.path.exists(MODEL_FILENAME):
        print(f"⚠️ PERINGATAN: File {MODEL_FILENAME} belum ada. Jalankan training dulu!")
    if not os.path.exists(VECTOR_FILENAME):
        print(f"⚠️ PERINGATAN: File {VECTOR_FILENAME} belum ada. Jalankan training dulu!")
        
    app.run(debug=True, host='0.0.0.0', port=5000)