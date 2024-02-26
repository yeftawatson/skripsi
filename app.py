from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from scipy.fftpack import dct
from tkinter import Tk, filedialog

app = Flask(__name__)

def dct_hash(image, hash_size=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct_result = dct(dct(gray_image, axis=0), axis=1)
    hash_code = ""
    for i in range(hash_size // 2):
        for j in range(hash_size // 2):
            hash_code += "1" if dct_result[i, j] > dct_result[i + 1, j + 1] else "0"         
    return hash_code

def hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        raise ValueError("Panjang hash tidak sama")
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

def pilih_gambar():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = 'static/' + file.filename
            file.save(file_path)
            return redirect(url_for('manipulate', filename=file.filename))

@app.route('/manipulate/<filename>')
def manipulate(filename):
    image_path = 'static/' + filename
    image = cv2.imread(image_path)
    manipulated_images = []
    factors = [2, 0.5, 50, 90]  # Skala untuk masing-masing manipulasi
    for factor in factors:
        if factor > 1:
            manipulated = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        elif factor < 1:
            manipulated = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        else:
            manipulated = np.clip(image + factor, 0, 255)
        cv2.imwrite(f'static/manipulated_{factor}.jpg', manipulated)
        manipulated_images.append(f'static/manipulated_{factor}.jpg')
    return render_template('result.html', filename=filename, manipulated_images=manipulated_images)

if __name__ == '__main__':
    app.run(debug=True)


#INI SOURCE PY
# import cv2
# import numpy as np
# from scipy.fftpack import dct
# from tkinter import Tk, filedialog


# def dct_hash(image, hash_size=8):
#     # Konversi gambar ke grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Lakukan DCT pada gambar
#     dct_result = dct(dct(gray_image, axis=0), axis=1)
    
#     # Ambil nilai di pojok kiri atas dari koefisien DCT
#     hash_code = ""
#     for i in range(hash_size // 2):
#         for j in range(hash_size // 2):
#             hash_code += "1" if dct_result[i, j] > dct_result[i + 1, j + 1] else "0"
                
#     return hash_code

# def hamming_distance(hash1, hash2):
#     # Pastikan panjang hash sama
#     if len(hash1) != len(hash2):
#         raise ValueError("Panjang hash tidak sama")
    
#     # Hitung jarak Hamming
#     distance = sum([char1 != char2 for char1, char2 in zip(hash1, hash2)])
#     return distance

# def authenticate_image(image1, image2, threshold):
#     hash1 = dct_hash(image1)
#     hash2 = dct_hash(image2)
#     print(hash1, hash2)
    
#     distance = hamming_distance(hash1, hash2)
    
#     if distance <= threshold:
#         return True
#     else:
#         return False
    
# # Fungsi untuk memilih gambar
# def pilih_gambar():
#     root = Tk()
#     root.withdraw()  # Sembunyikan jendela root
#     file_path = filedialog.askopenfilename()  # Buka jendela dialog pemilihan file
#     return file_path

# # Contoh penggunaan
# if __name__ == "__main__":
#     # Baca gambar
#     print("Silahkan Pilih Gambar yang terautentikasi")
#     gambar1 = pilih_gambar()
#     print("Silahkan Pilih Gambar yang akan di cek autentikasinya")
#     gambar2 = pilih_gambar()

#     image1 = cv2.imread(gambar1)
#     manipulated1 = image1.copy()
#     manipulated2 = image1.copy()
#     manipulated3 = image1.copy()
#     manipulated4 = image1.copy()
    
#     # nama = 1
#     # while nama != 0:
#     #     print("MENU MANIPULASI \n 1. Up Scaling \n 2. Down Scaling \n 3. Merubah Intensitas \n 4. Compress Size \n 0. Exit")
#     #     nama = int(input("Masukkan Pilihan: "))
#     #     if nama == 1:
#     #         faktor_skala = float(input("Masukkan Skala (Angka >1): "))
#     #         manipulated = cv2.resize(image2, None, fx=faktor_skala, fy=faktor_skala, interpolation=cv2.INTER_AREA)
#     #         break
#     #     elif nama == 2:
#     #         faktor_skala = float(input("Masukkan Skala (Angka <1): "))
#     #         manipulated = cv2.resize(image2, None, fx=faktor_skala, fy=faktor_skala, interpolation=cv2.INTER_AREA)
#     #         break
#     #     elif nama == 3:
#     #         gamma = int(input("Masukkan Skala (Contoh: 50): "))
#     #         manipulated = np.clip(image2 + gamma, 0, 255)
#     #         break
#     #     elif nama == 4:
#     #         compress = int(input("Masukkan Skala 0-100 (Higher scale, better image): "))
#     #         cv2.imwrite('gambar_kompres.jpg', image2, [cv2.IMWRITE_JPEG_QUALITY, compress])
#     #         break
#     #     elif nama == 0:
#     #         break
#     #     else:
#     #         print("Input Salah")
    
#     print("Upscaling")        
#     faktor_skala = float(input("Masukkan Skala (Angka >1): "))
#     manipulated1 = cv2.resize(image1, None, fx=faktor_skala, fy=faktor_skala, interpolation=cv2.INTER_AREA)

#     print("Downscaling")        
#     faktor_skala = float(input("Masukkan Skala (Angka <1): "))
#     manipulated2 = cv2.resize(image1, None, fx=faktor_skala, fy=faktor_skala, interpolation=cv2.INTER_AREA)
    
#     print("Perubahan Intensitas")        
#     gamma = int(input("Masukkan Skala (Contoh: 50): "))
#     manipulated3 = np.clip(image1 + gamma, 0, 255)
    
#     print("Compress File Size")
#     compress = int(input("Masukkan Skala 0-100 (Higher scale, better image): "))
#     cv2.imwrite('gambar_kompres.jpg', image1, [cv2.IMWRITE_JPEG_QUALITY, compress])
    

#     # Set ambang batas
#     threshold = 0.1
    
#     # Autentikasi gambar
#     is_authenticated1 = authenticate_image(image1, manipulated1, threshold)
#     is_authenticated2 = authenticate_image(image1, manipulated2, threshold)
#     is_authenticated3 = authenticate_image(image1, manipulated3, threshold)
#     is_authenticated4 = authenticate_image(image1, manipulated4, threshold)
    
#     # Output hasil autentikasi
#     if is_authenticated1:
#         print("Gambar terautentikasi.")
#     else:
#         print("Gambar tidak terautentikasi.")
        
#     if is_authenticated2:
#         print("Gambar terautentikasi.")
#     else:
#         print("Gambar tidak terautentikasi.")
        
#     if is_authenticated3:
#         print("Gambar terautentikasi.")
#     else:
#         print("Gambar tidak terautentikasi.")
        
#     if is_authenticated4:
#         print("Gambar terautentikasi.")
#     else:
#         print("Gambar tidak terautentikasi.")
