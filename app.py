import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình đã huấn luyện
model = load_model('rubbish_classification_model.h5')

# Định nghĩa kích thước ảnh
IMG_SIZE = (150, 150)

# Tạo thư mục tạm thời nếu chưa tồn tại
if not os.path.exists('temp'):
    os.makedirs('temp')

# Route cho trang chủ
@app.route('/')
def home():
    return render_template('index.html')  # Trả về file index.html

# Route để nhận diện rác thải
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='No file provided')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', result='No file selected')

    # Lưu file vào một vị trí tạm thời
    temp_file_path = os.path.join('temp', file.filename)
    file.save(temp_file_path)

    # Xử lý ảnh
    img = load_img(temp_file_path, target_size=IMG_SIZE)  # Sử dụng đường dẫn tạm thời
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    prediction = model.predict(img_array)
    class_label = 'Recyclable' if prediction[0] > 0.5 else 'Non-Recyclable'

    # Xóa file tạm thời
    os.remove(temp_file_path)

    return render_template('index.html', result=class_label)  # Trả về kết quả dự đoán

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
