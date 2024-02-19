import cv2
import numpy as np
from scipy.fftpack import dct

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

# Contoh penggunaan
if __name__ == "__main__":
    # Baca gambar
    image1 = cv2.imread("gambar1.png")
    image2 = cv2.imread("gambar2.png")
    
    # Set ambang batas
    threshold = 10
    
    # Autentikasi gambar
    is_authenticated = authenticate_image(image1, image2, threshold)
    
    # Output hasil autentikasi
    if is_authenticated:
        print("Gambar terautentikasi.")
    else:
        print("Gambar tidak terautentikasi.")
