import os

# Ganti path ini sesuai lokasi foldermu
path_dataset = "dataset_ready/test" 


expected_classes = [
    'cercospora', 
    'karat_daun', 
    'phoma', 
    'sehat'
]

print(f"Mengecek lokasi: {path_dataset}\n")

found_folders = os.listdir(path_dataset)
print(f"Folder yang ditemukan ({len(found_folders)}):")
print(found_folders)

print("\n--- HASIL PENGECEKAN ---")
for cls in expected_classes:
    if cls in found_folders:
        isi = os.listdir(os.path.join(path_dataset, cls))
        count = len(isi)
        status = "✅ OKE" if count > 10 else "⚠️ TERLALU SEDIKIT"
        print(f"{status} - {cls}: {count} gambar")
    else:
        print(f"❌ HILANG - {cls} (Folder tidak ditemukan!)")