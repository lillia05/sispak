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
    Implementasi Presisi R-01 s/d R-20
    Menggunakan Logika Threshold (Ambang Batas) untuk IF-THEN rules.
    """
    scores = {'karat_daun': 0.0, 'cercospora': 0.0, 'phoma': 0.0, 'sehat': 0.0}

    # --- DEFINISI THRESHOLD (BATAS NILAI) ---
    TH_TINGGI = 0.7  # Mewakili: Sangat Jelas, Pekat, Besar, Rusak Parah
    TH_SEDANG = 0.4  # Mewakili: Cukup, Sedang, Terlihat
    TH_RENDAH = 0.2  # Mewakili: Sedikit, Samar, Tipis

    # --- MAPPING INPUT ---
    bubuk    = i['tekstur_bubuk']
    oranye   = i['warna_oranye']
    putih    = i['pusat_putih']
    halo     = i['halo_kuning']
    hitam    = i['bercak_hitam']
    keriting = i['pinggiran_keriting']
    pucuk    = i['posisi_pucuk']

    # 1. PENYAKIT KARAT DAUN (Leaf Rust)
    
    # R-01: IF Tekstur_Bubuk Tinggi AND Warna_Oranye Tinggi
    if bubuk >= TH_TINGGI and oranye >= TH_TINGGI:
        scores['karat_daun'] = max(scores['karat_daun'], 1.0)

    # R-02: IF (Bubuk Sedang OR Tinggi) AND Warna_Oranye Sedang
    if bubuk >= TH_SEDANG and oranye >= TH_SEDANG:
        scores['karat_daun'] = max(scores['karat_daun'], 0.8)

    # R-03: IF Oranye Tinggi AND Bubuk Rendah AND Hitam Rendah (Fase Awal)
    if (oranye >= TH_TINGGI) and (bubuk < TH_SEDANG) and (hitam < TH_RENDAH):
        scores['karat_daun'] = max(scores['karat_daun'], 0.6)

    # R-04: IF (Oranye Sedang OR Tinggi) AND Putih Rendah AND Pucuk Rendah
    if (oranye >= TH_SEDANG) and (putih < TH_RENDAH) and (pucuk < TH_RENDAH):
        scores['karat_daun'] = max(scores['karat_daun'], 0.8)

    # --- RULES TAMBAHAN (PEMBEDA) ---
    
    # R-16: IF Oranye Jelas AND Halo Tidak Ada AND Putih Tidak Ada
    if (oranye >= TH_TINGGI) and (halo < TH_RENDAH) and (putih < TH_RENDAH):
        scores['karat_daun'] = max(scores['karat_daun'], 1.0)

    # R-17: IF Bubuk Sedikit (>= Rendah) AND Putih Tidak Ada
    if (bubuk >= TH_RENDAH) and (putih < TH_RENDAH):
        scores['karat_daun'] = max(scores['karat_daun'], 0.7)

    # 2. PENYAKIT BERCAK DAUN CERCOSPORA (Mata Ayam)

    # R-05: IF Pusat_Putih Tinggi AND Halo_Kuning Tinggi
    if putih >= TH_TINGGI and halo >= TH_TINGGI:
        scores['cercospora'] = max(scores['cercospora'], 1.0)

    # R-06: IF (Putih Sedang OR Tinggi) AND Halo_Kuning Sedang
    if putih >= TH_SEDANG and halo >= TH_SEDANG:
        scores['cercospora'] = max(scores['cercospora'], 0.8)

    # R-07: IF Putih Tinggi AND Halo Rendah AND Keriting Rendah
    if (putih >= TH_TINGGI) and (halo < TH_SEDANG) and (keriting < TH_RENDAH):
        scores['cercospora'] = max(scores['cercospora'], 0.6)

    # R-08: IF (Putih Sedang OR Halo Sedang) AND Oranye Rendah AND Hitam Rendah
    if (putih >= TH_SEDANG or halo >= TH_SEDANG) and (oranye < TH_RENDAH) and (hitam < TH_RENDAH):
        scores['cercospora'] = max(scores['cercospora'], 0.5)

    # --- RULES TAMBAHAN (PEMBEDA) ---

    # R-14: IF Halo Jelas AND Bubuk Tidak Ada
    if (halo >= TH_TINGGI) and (bubuk < TH_RENDAH):
        scores['cercospora'] = max(scores['cercospora'], 1.0)

    # R-15: IF Halo Jelas AND Oranye Tidak Ada
    if (halo >= TH_TINGGI) and (oranye < TH_RENDAH):
        scores['cercospora'] = max(scores['cercospora'], 0.8)

    # R-18: IF Putih Jelas AND Pinggiran Normal (Tidak Keriting)
    if (putih >= TH_TINGGI) and (keriting < TH_RENDAH):
        scores['cercospora'] = max(scores['cercospora'], 0.8)
    
    # 3. PENYAKIT PHOMA (American Leaf Spot)    

    # R-09: IF Bercak_Hitam Tinggi AND Pinggiran_Keriting Tinggi
    if hitam >= TH_TINGGI and keriting >= TH_TINGGI:
        scores['phoma'] = max(scores['phoma'], 1.0)

    # R-10: IF (Pucuk Sedang OR Tinggi) AND Pinggiran_Keriting Tinggi
    if pucuk >= TH_SEDANG and keriting >= TH_TINGGI:
        scores['phoma'] = max(scores['phoma'], 0.8)

    # R-11: IF Bercak_Hitam Tinggi AND Pucuk Tinggi AND Pusat_Putih Rendah
    if (hitam >= TH_TINGGI) and (pucuk >= TH_TINGGI) and (putih < TH_RENDAH):
        scores['phoma'] = max(scores['phoma'], 0.8)

    # R-12: IF (Hitam Sedang OR Tinggi) AND (Keriting Sedang OR Tinggi) AND Oranye Rendah
    if (hitam >= TH_SEDANG) and (keriting >= TH_SEDANG) and (oranye < TH_RENDAH):
        scores['phoma'] = max(scores['phoma'], 0.6)

    # --- RULES TAMBAHAN (PEMBEDA) ---

    # R-19: IF Hitam Tinggi AND Keriting Cukup (Sedang) AND Putih Tidak Ada
    if (hitam >= TH_TINGGI) and (keriting >= TH_SEDANG) and (putih < TH_RENDAH):
        scores['phoma'] = max(scores['phoma'], 1.0)
    
    # R-20: IF Keriting Sedikit (>= Rendah) AND Putih Samar (>= Rendah tapi < Tinggi)
    # Prioritas Phoma Sedang, Cercospora Rendah
    if (keriting >= TH_RENDAH) and (putih >= TH_RENDAH and putih < TH_TINGGI):
        scores['phoma'] = max(scores['phoma'], 0.6)      # Phoma Sedang
        scores['cercospora'] = min(scores['cercospora'], 0.4) # Cercospora Rendah/Turun

    # 4. KONDISI SEHAT    
    # R-13: IF Semua Gejala Rendah (Tidak Ada)
    if (bubuk < TH_RENDAH) and (oranye < TH_RENDAH) and (putih < TH_RENDAH) and \
       (hitam < TH_RENDAH) and (keriting < TH_RENDAH):
        scores['sehat'] = 1.0
    else:
        # Fallback logic jika tidak sehat tapi tidak ada penyakit yang kuat
        if max(scores.values()) < 0.3:
            scores['sehat'] = 0.5 # Kemungkinan masalah abiotik/bukan penyakit di database

    return scores
if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)