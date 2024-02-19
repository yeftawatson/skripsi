import cv2
import numpy as np
from scipy.fftpack import dct
from tkinter import Tk, filedialog


def dct_hash(image, hash_size=8):
    # Konversi gambar ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize gambar ke ukuran yang diinginkan
    resized_image = cv2.resize(gray_image, (hash_size, hash_size))
    
    # Lakukan DCT pada gambar
    dct_result = dct(dct(resized_image, axis=0), axis=1)
    
    # Ambil nilai di pojok kiri atas dari koefisien DCT
    hash_code = ""
    for i in range(hash_size // 2):
        for j in range(hash_size // 2):
            hash_code += "1" if dct_result[i, j] > dct_result[i + 1, j + 1] else "0"
                
    return hash_code

def hamming_distance(hash1, hash2):
    # Pastikan panjang hash sama
    if len(hash1) != len(hash2):
        raise ValueError("Panjang hash tidak sama")
    
    # Hitung jarak Hamming
    distance = sum([char1 != char2 for char1, char2 in zip(hash1, hash2)])
    return distance

def authenticate_image(image1, image2, threshold):
    hash1 = dct_hash(image1)
    hash2 = dct_hash(image2)
    
    distance = hamming_distance(hash1, hash2)
    
    if distance <= threshold:
        return True
    else:
        return False
    
# Fungsi untuk memilih gambar
def pilih_gambar():
    root = Tk()
    root.withdraw()  # Sembunyikan jendela root
    file_path = filedialog.askopenfilename()  # Buka jendela dialog pemilihan file
    return file_path

# Contoh penggunaan
if __name__ == "__main__":
    # Baca gambar
    print("Silahkan Pilih Gambar yang terautentikasi")
    gambar1 = pilih_gambar()
    print("Silahkan Pilih Gambar yang akan di cek autentikasinya")
    gambar2 = pilih_gambar()

    image1 = cv2.imread(gambar1)
    image2 = cv2.imread(gambar2)
    manipulated = image2.copy()
    
    nama = 1
    while nama != 0:
        print("MENU MANIPULASI \n 1. Up Scaling \n 2. Down Scaling \n 3. Merubah Intensitas \n 4. Compress Size \n 0. Exit")
        nama = int(input("Masukkan Pilihan: "))
        if nama == 1:
            faktor_skala = float(input("Masukkan Skala (Angka >1): "))
            manipulated = cv2.resize(image2, None, fx=faktor_skala, fy=faktor_skala, interpolation=cv2.INTER_AREA)
            break
        elif nama == 2:
            faktor_skala = float(input("Masukkan Skala (Angka <1): "))
            manipulated = cv2.resize(image2, None, fx=faktor_skala, fy=faktor_skala, interpolation=cv2.INTER_AREA)
            break
        elif nama == 3:
            gamma = int(input("Masukkan Skala (Contoh: 50): "))
            manipulated = np.clip(image2 + gamma, 0, 255)
            break
        elif nama == 4:
            compress = int(input("Masukkan Skala 0-100 (Higher scale, better image): "))
            cv2.imwrite('gambar_kompres.jpg', image2, [cv2.IMWRITE_JPEG_QUALITY, compress])
            break
        elif nama == 0:
            break
        else:
            print("Input Salah")

    # Set ambang batas
    threshold = 10
    
    # Autentikasi gambar
    is_authenticated = authenticate_image(image1, manipulated, threshold)
    
    # Output hasil autentikasi
    if is_authenticated:
        print("Gambar terautentikasi.")
    else:
        print("Gambar tidak terautentikasi.")
