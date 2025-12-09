from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max 5MB

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
MODEL_FILENAME = 'best_coffee_mobilenet_v2-new.h5'
CLASS_NAMES = ['cercospora', 'karat_daun', 'phoma', 'sehat']

# Database Penyakit
DISEASE_INFO = {
    'cercospora': {
        'name': 'Cercospora (Mata Ayam)',
        'description': 'Bercak daun dengan pusat berwarna abu-abu/putih (mata) dikelilingi halo kuning.',
        'treatment': 'Pangkas daun sakit, atur naungan agar tidak terlalu lembab, gunakan fungisida tembaga.'
    },
    'karat_daun': {
        'name': 'Karat Daun (Leaf Rust)',
        'description': 'Terdapat serbuk/tepung berwarna oranye jingga di permukaan bawah daun.',
        'treatment': 'Gunakan varietas tahan karat, pemupukan seimbang, dan fungisida sistemik jika parah.'
    },
    'phoma': {
        'name': 'Phoma (American Leaf Spot)',
        'description': 'Bercak hitam pekat tidak beraturan, seringkali membuat pinggiran daun keriting/gosong.',
        'treatment': 'Kurangi kelembaban, pangkas tunas air, semprot fungisida saat pembentukan buah.'
    },
    'sehat': {
        'name': 'Tanaman Sehat',
        'description': 'Daun hijau segar, tidak ada bercak, serbuk, atau kerusakan fisik.',
        'treatment': 'Pertahankan perawatan rutin (pupuk & air) dan monitoring berkala.'
    }
}

# Bobot Hybrid (Sesuai Request: 30% AI, 70% Rules)
ALPHA = 0.3  # Bobot AI
BETA = 0.7   # Bobot Sistem Pakar

# ==================== LOAD MODEL ====================
model = None
def load_model():
    global model
    if model is None and os.path.exists(MODEL_FILENAME):
        model = keras.models.load_model(MODEL_FILENAME)
        print("✅ Model berhasil dimuat")
    elif not os.path.exists(MODEL_FILENAME):
        print("⚠️ Model tidak ditemukan.")

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
            return jsonify({'error': 'Model belum dimuat'}), 500

        # --- 1. PROSES GAMBAR (AI) ---
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada gambar'}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.expand_dims(preprocess_input(np.array(img_resized)), axis=0)
        
        # Prediksi AI
        ai_probs = model.predict(img_arr, verbose=0)[0]

        # --- 2. PROSES GEJALA (SISTEM PAKAR) ---
        # Ambil input spesifik dari form (Nilai 0.0 s/d 1.0)
        inputs = {
            'tekstur_bubuk': float(request.form.get('tekstur_bubuk', 0)),
            'warna_oranye': float(request.form.get('warna_oranye', 0)),
            'pusat_putih': float(request.form.get('pusat_putih', 0)),
            'halo_kuning': float(request.form.get('halo_kuning', 0)),
            'bercak_hitam': float(request.form.get('bercak_hitam', 0)),
            'pinggiran_keriting': float(request.form.get('pinggiran_keriting', 0)),
            'posisi_pucuk': float(request.form.get('posisi_pucuk', 0))
        }

        manual_scores = calculate_expert_rules(inputs)

        # --- 3. HITUNG SKOR AKHIR (HYBRID) ---
        final_scores = {}
        for i, disease in enumerate(CLASS_NAMES):
            ai_score = float(ai_probs[i])
            manual_score = manual_scores.get(disease, 0.0)
            
            # Rumus: (AI * 0.3) + (Manual * 0.7)
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
        return jsonify({'error': str(e)}), 500

# ==================== RULE BASE (R-01 s/d R-20) ====================
def calculate_expert_rules(i):
    """
    Implementasi Rule Base menggunakan Logika Fuzzy sederhana.
    Min = AND, Max = OR.
    """
    # Inisialisasi skor penyakit
    scores = {'karat_daun': 0.0, 'cercospora': 0.0, 'phoma': 0.0, 'sehat': 0.0}
    
    # Helper variables (aliases agar koding lebih pendek)
    bubuk = i['tekstur_bubuk']
    oranye = i['warna_oranye']
    putih = i['pusat_putih']
    halo = i['halo_kuning']
    hitam = i['bercak_hitam']
    keriting = i['pinggiran_keriting']
    pucuk = i['posisi_pucuk']

    # --- PENYAKIT KARAT DAUN (Leaf Rust) ---
    # R-01: IF Bubuk Tinggi AND Oranye Tinggi
    r01 = min(bubuk, oranye)
    # R-02: IF (Bubuk Sedang/Tinggi) AND Oranye Sedang
    r02 = min(bubuk, max(0.5, oranye)) * 0.8 
    # R-03: Oranye Tinggi, Bubuk Rendah, Hitam Rendah (Fase Awal)
    r03 = min(oranye, (1-bubuk), (1-hitam)) * 0.6
    # R-04: Oranye Ada, Putih Rendah, Pucuk Rendah
    r04 = min(oranye, (1-putih), (1-pucuk)) * 0.8
    # R-16 & R-17 (Rules Tambahan Pembeda)
    r16 = min(oranye, (1-halo), (1-putih))
    r17 = min(bubuk, (1-putih)) * 0.7

    scores['karat_daun'] = max(r01, r02, r03, r04, r16, r17)

    # --- PENYAKIT CERCOSPORA (Mata Ayam) ---
    # R-05: IF Pusat Putih Tinggi AND Halo Kuning Tinggi
    r05 = min(putih, halo)
    # R-06: IF (Putih Ada) AND Halo Sedang
    r06 = min(putih, halo) * 0.8
    # R-07: Putih Tinggi, Halo Rendah, Keriting Rendah
    r07 = min(putih, (1-halo), (1-keriting)) * 0.7
    # R-08: Gejala Samar tapi bukan Oranye/Hitam
    r08 = min(max(putih, halo), (1-oranye), (1-hitam)) * 0.5
    # R-14 & R-15 (Pembeda Karat)
    r14 = min(halo, (1-bubuk))
    r15 = min(halo, (1-oranye)) * 0.9
    # R-18 (Pembeda Phoma: Putih Jelas + Daun Datar)
    r18 = min(putih, (1-keriting))

    scores['cercospora'] = max(r05, r06, r07, r08, r14, r15, r18)

    # --- PENYAKIT PHOMA (Bercak Hitam) ---
    # R-09: IF Hitam Tinggi AND Keriting Tinggi
    r09 = min(hitam, keriting)
    # R-10: Pucuk Tinggi AND Keriting Tinggi
    r10 = min(pucuk, keriting) * 0.9
    # R-11: Hitam Tinggi AND Pucuk Tinggi AND Putih Rendah
    r11 = min(hitam, pucuk, (1-putih))
    # R-12: Hitam/Keriting Ada AND Oranye Rendah
    r12 = min(max(hitam, keriting), (1-oranye)) * 0.6
    # R-19: Hitam + Keriting + Tidak Putih
    r19 = min(hitam, keriting, (1-putih))
    # R-20: Keriting Sedikit + Putih Samar -> Prioritas Phoma
    r20 = min(keriting * 0.5, putih * 0.3) 

    scores['phoma'] = max(r09, r10, r11, r12, r19, r20)

    # --- KONDISI SEHAT ---
    # R-13: Semua gejala rendah
    r13 = min((1-bubuk), (1-oranye), (1-putih), (1-hitam), (1-keriting))
    scores['sehat'] = r13

    return scores

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)