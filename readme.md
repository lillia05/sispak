# ☕ SISPAKO – Sistem Pakar Deteksi Penyakit Daun Kopi  
SISPAKO adalah aplikasi berbasis **Python + Flask** yang digunakan untuk mendeteksi penyakit daun kopi seperti Karat Daun, Cercospora, Phoma, dan lainnya.  
Sistem ini juga dilengkapi antarmuka web yang modern dan responsif.

---

## 🚀 Fitur Utama
- Deteksi penyakit daun kopi berbasis **Logika Fuzzy / Machine Learning**.
- Tampilan web modern dengan **Bootstrap 5**.
- Halaman:
  - Beranda
  - Info Penyakit
  - Diagnosa
  - Tentang
- Mendukung Upload Gambar & Analisis Otomatis.
- API endpoint Flask untuk prediksi.

---

## 📦 1. Persyaratan Sistem
Pastikan perangkat memiliki:

- **Python 3.10+**
- **Pip** (sudah include di Python)
- **Virtual environment** (opsional, tapi disarankan)
- **Browser modern** (Chrome / Firefox)

---

## 📥 2. Cara Instalasi

### 2.1 Clone / Download Project
```bash
git clone https://github.com/username/sispako.git
cd sispako

# Membuat Virtual Environment (Opsional tapi Disarankan)

python -m venv venv
venv\Scripts\activate

# Install Dependency

pip install flask
pip install numpy
pip install tensorflow
pip install pillow
pip install scikit-learn

# Struktur Project

project/
│-- app.py
│-- requirements.txt
│-- static/
│   └── img/
│-- templates/
│   ├── index.html
│   ├── diagnosa.html
│   ├── penyakit.html
│   ├── tentang.html
│   └── components/
│       └── navbar.html
└── model/
    └── model.h5 (opsional)

# Cara menjalankan

python app.py

# CARA MENGGUNAKAN ENGINE PERTAMA KALI

1. extract file data mentah tapi full
2. buat folder kosong bernama dataset_ready
3. jalankan split_dataset.py -> python split_dataset.py 
4. lalu jalankan engine.py -> python engine.py
5. pilih opsi 2 ( TRAIN )
6. tunggu hingga selesai 
7. pilih 3
8. masukan path gambar tanpa " ".

Contoh -> D:\Kuliah Unila\Semester 5\Sistem Pakar\User Interface\static\img\riskur.jpeg