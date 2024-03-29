from flask import Flask, render_template, request
import cv2
import numpy as np
from scipy.fftpack import dct
import os
import pywt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def dct_hash(image, hash_size=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct_result = dct(dct(gray_image, axis=0), axis=1)
    hash_code = ""
    for i in range(hash_size):
        for j in range(hash_size):
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

def wavelet_hash(image, hash_size=8):
    coeffs = pywt.dwt2(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 'haar')
    LL, (LH, HL, HH) = coeffs
    hash_code = ""
    for i in range(hash_size):
        for j in range(hash_size):
            hash_code += "1" if LH[i, j] > LH[i + 1, j + 1] else "0"
    return hash_code

def authenticate_wavelet_hash(image1, image2, threshold):
    hash1 = wavelet_hash(image1)
    hash2 = wavelet_hash(image2)
    distance = hamming_distance(hash1, hash2)
    if distance <= threshold:
        return True
    else:
        return False

def average_hash(image, hash_size=8):
    resized_image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    average_value = np.mean(gray_image)
    hash_code = ""
    for i in range(hash_size):
        for j in range(hash_size):
            hash_code += "1" if gray_image[i, j] > average_value else "0"
    return hash_code

def authenticate_average_hash(image1, image2, threshold):
    hash1 = average_hash(image1)
    hash2 = average_hash(image2)
    distance = hamming_distance(hash1, hash2)
    if distance <= threshold:
        return True
    else:
        return False
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='Tidak ada file yang dipilih')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='Tidak ada file yang dipilih')
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Baca gambar yang diupload
            image1 = cv2.imread(filepath)

            # Skala yang diinputkan melalui website
            faktor_skala_up = float(request.form['faktor_skala_up'])
            faktor_skala_down = float(request.form['faktor_skala_down'])
            gamma = int(request.form['gamma'])
            compress_scale = int(request.form['compress_scale'])

            manipulated_images = []
            manipulated_images.append(cv2.resize(image1, None, fx=faktor_skala_up, fy=faktor_skala_up, interpolation=cv2.INTER_AREA))
            manipulated_images.append(cv2.resize(image1, None, fx=faktor_skala_down, fy=faktor_skala_down, interpolation=cv2.INTER_AREA))
            manipulated_images.append(np.clip(image1 + gamma, 0, 255))
            
            # Mengompresi gambar
            compress_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_image.jpg')
            cv2.imwrite(compress_filepath, image1, [cv2.IMWRITE_JPEG_QUALITY, compress_scale])
            manipulated_images.append(cv2.imread(compress_filepath))

            # Simpan gambar yang dimanipulasi
            manipulated_image_paths = []
            for idx, manipulated_image in enumerate(manipulated_images):
                manipulated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'manipulated_image_{idx}.jpg')
                cv2.imwrite(manipulated_image_path, manipulated_image)
                manipulated_image_paths.append(manipulated_image_path)

            # Set ambang batas
            threshold = 8 #0.125

            # Autentikasi gambar
            # Autentikasi gambar
            results_dct = []
            results_wavelet = []
            results_average = []
            for manipulated_image_path in manipulated_image_paths:
                manipulated_image = cv2.imread(manipulated_image_path)
                is_authenticated_dct = authenticate_image(image1, manipulated_image, threshold)
                is_authenticated_wavelet = authenticate_wavelet_hash(image1, manipulated_image, threshold)
                is_authenticated_average = authenticate_average_hash(image1, manipulated_image, threshold)
                results_dct.append(is_authenticated_dct)
                results_wavelet.append(is_authenticated_wavelet)
                results_average.append(is_authenticated_average)

            # Ubah path gambar menjadi URL
            original_image = '/' + filepath
            manipulated_image_paths = ['/' + path for path in manipulated_image_paths]

            return render_template('index.html', message='File berhasil diupload', original_image=original_image, manipulated_image_paths=manipulated_image_paths, results_dct=results_dct, results_wavelet=results_wavelet, results_average=results_average)


    return render_template('index.html')


if __name__ == '__main__': 
    app.run(debug=True)
