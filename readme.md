# ☕ SISPAKO – Sistem Pakar Deteksi Penyakit Daun Kopi  
SISPAKO adalah aplikasi berbasis **Python + Flask** yang digunakan untuk mendeteksi penyakit daun kopi seperti Karat Daun, Cercospora, Phoma, dan lainnya.  
Sistem ini juga dilengkapi antarmuka web yang modern dan responsif.

---

## 👥 Anggota Tim Pengembang

1. **Nadya Arsa** (2317051033)
2. **Lekok Indah Lia** (2317051097)
3. **Rizky Kurnia Antasari** (2357051011)

---

## 📖 Tentang Proyek

Sistem ini dirancang untuk membantu petani kopi mengidentifikasi penyakit daun secara dini. Keunggulan sistem ini terletak pada metode **Hybrid Decision Support**, di mana diagnosa tidak hanya bergantung pada foto (yang rentan bias cahaya), tetapi juga divalidasi dengan gejala fisik yang diamati pengguna.

### 🧠 Metode Hybrid (Bobot 30:70)
1.  **Computer Vision (Bobot 30%):** Menggunakan model **MobileNetV2** yang telah dilatih (*fine-tuning*) untuk mengenali pola visual penyakit dari citra daun.
2.  **Sistem Pakar Fuzzy (Bobot 70%):** Menggunakan logika **Fuzzy Mamdani** dengan inferensi *Forward Chaining* berdasarkan 7 gejala fisik dan 20 aturan (*rules*) pakar.

### 🦠 Penyakit yang Dideteksi
* **Karat Daun** (*Coffee Leaf Rust*)
* **Cercospora** (*Brown Eye Spot*)
* **Phoma** (*American Leaf Spot*)
* **Tanaman Sehat**

---
## ✨ Fitur Utama

* **Diagnosa Hybrid:** Menggabungkan probabilitas CNN dan skor Fuzzy untuk akurasi tinggi.
* **Input Ganda:** Upload foto daun & Kuesioner 7 gejala visual interaktif.
* **Validasi Keamanan:** Dilengkapi fitur *Similarity Check* (Cosine Similarity) untuk menolak gambar yang bukan daun kopi.
* **Edukasi Penyakit:** Menyediakan informasi detail mengenai ciri-ciri penyakit dan solusi penanganan (Kultur Teknis & Kimiawi).
* **Antarmuka Responsif:** Dibangun dengan Bootstrap 5, nyaman diakses via HP maupun Desktop.
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
pip install opencv-python

# Cara menjalankan
python app.py


# Struktur Project
sispak/
│
├── app.py                       # Main application logic (Flask + AI + Fuzzy)
├── best_coffee_model.h5         # Trained Model MobileNetV2
├── coffee_reference_vector.npy   # Reference Vector for Similarity Check
├── requirements.txt             # List dependencies
│
├── static/
│   ├── css/                     # Stylesheets
│   ├── img/                     # Asset gambar (logo, sampel penyakit)
│   └── js/                      # Scripts
│
├── templates/
│   ├── components/              # Navbar & Footer partials
│   ├── index.html               # Halaman Beranda
│   ├── diagnosa.html            # Halaman Input Diagnosa
│   ├── hasil_diagnosa.html       # Halaman Hasil (Result)
│   ├── penyakit.html            # Halaman Info Penyakit
│   └── tentang.html             # Halaman Tentang Tim
│
└── README.md                    # Dokumentasi Proyek


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
