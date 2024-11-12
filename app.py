import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io
import cv2

app = Flask(__name__)

# Tải mô hình đã huấn luyện
model = load_model('rubbish_classification_model.h5')
IMG_SIZE = (150, 150)

@app.route('/')
def home():
    return render_template('index.html')

# Route để nhận dạng ảnh upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='No file provided')

    file = request.files['file']
    img = load_img(file, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán phân loại
    prediction = model.predict(img_array)
    class_label = 'Recyclable' if prediction[0] > 0.5 else 'Non-Recyclable'

    return render_template('index.html', result=class_label)

# Route để nhận ảnh từ camera và phân loại
@app.route('/predict-camera', methods=['POST'])
def predict_camera():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(',')[1]  # Lấy phần base64 của ảnh

    # Chuyển đổi base64 thành ảnh
    img_data = base64.b64decode(image_data)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán phân loại
    prediction = model.predict(img_array)
    class_label = 'Recyclable' if prediction[0] > 0.5 else 'Non-Recyclable'

    return jsonify({ 'result': class_label })

if __name__ == '__main__':
    app.run(debug=True)
