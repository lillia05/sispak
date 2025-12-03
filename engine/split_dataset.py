import os
import shutil
import random
import glob

# ================= KONFIGURASI =================
# 1. Masukkan nama folder tempat kamu menyimpan 100% gambar mentah
SOURCE_FOLDER = "dataset_organized" 

# 2. Nama folder hasil output (yang strukturnya seperti screenshot kamu)
OUTPUT_FOLDER = "dataset_ready"

# 3. Persentase Split (Total harus 1.0 atau 100%)
TRAIN_RATIO = 0.70  # 70%
VAL_RATIO   = 0.15  # 15%
TEST_RATIO  = 0.15  # 15%
# ===============================================

def split_data():
    # Cek folder sumber
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ Error: Folder '{SOURCE_FOLDER}' tidak ditemukan!")
        print("👉 Buat folder baru bernama 'dataset_sumber', lalu masukkan folder kelas-kelas (cercospora, dll) yang berisi 100% gambar di situ.")
        return

    # Reset folder output biar bersih
    if os.path.exists(OUTPUT_FOLDER):
        print(f"♻️  Membersihkan folder '{OUTPUT_FOLDER}' lama...")
        shutil.rmtree(OUTPUT_FOLDER)

    # Ambil daftar nama kelas (nama folder di dalam source)
    classes = [d for d in os.listdir(SOURCE_FOLDER) if os.path.isdir(os.path.join(SOURCE_FOLDER, d))]
    
    if not classes:
        print("❌ Tidak ada folder kelas ditemukan di dalam source!")
        return

    print(f"✅ Ditemukan kelas: {classes}")
    print("🚀 Memulai proses pembagian (70% Train, 15% Val, 15% Test)...\n")

    total_images_processed = 0

    for class_name in classes:
        # Path ke folder kelas asli (100% data)
        class_dir = os.path.join(SOURCE_FOLDER, class_name)
        
        # Ambil semua gambar (jpg, png, jpeg)
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(glob.glob(os.path.join(class_dir, ext)))
        
        # ACAK URUTAN GAMBAR (Penting biar splitnya adil)
        random.shuffle(images)
        
        total = len(images)
        if total == 0:
            print(f"⚠️  Kelas '{class_name}' kosong! Dilewati.")
            continue

        # Hitung Matematika Pembagiannya
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        test_count = total - train_count - val_count # Sisanya masuk test

        # Potong List Gambar
        train_imgs = images[:train_count]
        val_imgs = images[train_count : train_count + val_count]
        test_imgs = images[train_count + val_count :]

        # Fungsi helper untuk copy file
        def copy_files(file_list, split_type):
            dest_dir = os.path.join(OUTPUT_FOLDER, split_type, class_name)
            os.makedirs(dest_dir, exist_ok=True) # Buat folder otomatis
            
            for img_path in file_list:
                fname = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(dest_dir, fname))

        # Eksekusi Copy
        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_images = copy_files(test_imgs, 'test')

        print(f"📂 Class: {class_name} (Total: {total})")
        print(f"   pool -> Train : {len(train_imgs)} gambar")
        print(f"   pool -> Val   : {len(val_imgs)} gambar")
        print(f"   pool -> Test  : {len(test_imgs)} gambar")
        
        total_images_processed += total

    print("\n" + "="*50)
    print(f"🎉 SELESAI! Struktur folder {OUTPUT_FOLDER} sudah siap.")
    print(f"Total gambar diproses: {total_images_processed}")
    print("="*50)

if __name__ == "__main__":
    split_data()