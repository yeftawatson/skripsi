from flask import Flask, render_template, request
import cv2
import numpy as np
from scipy.fftpack import dct
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
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

            manipulated_images = []
            manipulated_images.append(cv2.resize(image1, None, fx=faktor_skala_up, fy=faktor_skala_up, interpolation=cv2.INTER_AREA))
            manipulated_images.append(cv2.resize(image1, None, fx=faktor_skala_down, fy=faktor_skala_down, interpolation=cv2.INTER_AREA))
            manipulated_images.append(np.clip(image1 + gamma, 0, 255))

            # Simpan gambar yang dimanipulasi
            manipulated_image_paths = []
            for idx, manipulated_image in enumerate(manipulated_images):
                manipulated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'manipulated_image_{idx}.jpg')
                cv2.imwrite(manipulated_image_path, manipulated_image)
                manipulated_image_paths.append(manipulated_image_path)

            # Set ambang batas
            threshold = 0.1

            # Autentikasi gambar
            results = []
            for manipulated_image_path in manipulated_image_paths:
                manipulated_image = cv2.imread(manipulated_image_path)
                is_authenticated = authenticate_image(image1, manipulated_image, threshold)
                results.append(is_authenticated)

            return render_template('index.html', message='File berhasil diupload', original_image=filepath, manipulated_image_paths=manipulated_image_paths, results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
