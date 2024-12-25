import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import base64
import sqlite3
from io import BytesIO
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Đặt secret key cho session

# Tải mô hình đã huấn luyện
model = load_model('rubbish_classification_model.h5')
IMG_SIZE = (150, 150)


# Kết nối và tạo bảng lưu trữ kết quả và người dùng
def init_db():
    conn = sqlite3.connect('rubbish_classification.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


# Gọi hàm tạo bảng khi server khởi động
init_db()


# Hàm lưu kết quả vào cơ sở dữ liệu
def log_classification(label):
    conn = sqlite3.connect('rubbish_classification.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO classification_logs (label, timestamp) VALUES (?, ?)',
                   (label, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()


# Hàm lấy thống kê dữ liệu
def get_statistics():
    conn = sqlite3.connect('rubbish_classification.db')
    cursor = conn.cursor()
    cursor.execute('SELECT label, COUNT(*) as count FROM classification_logs GROUP BY label')
    stats = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in stats}


@app.route('/')
def home():
    if 'user_id' not in session:
        flash('Vui lòng đăng nhập để truy cập trang chính.', 'error')
        return redirect(url_for('login'))
    stats = get_statistics()
    recyclable_count = stats.get('Recyclable', 0)
    non_recyclable_count = stats.get('Non-Recyclable', 0)
    total = recyclable_count + non_recyclable_count
    recyclable_percentage = (recyclable_count / total * 100) if total > 0 else 0
    non_recyclable_percentage = (non_recyclable_count / total * 100) if total > 0 else 0

    return render_template(
        'index.html',
        recyclable_count=recyclable_count,
        non_recyclable_count=non_recyclable_count,
        recyclable_percentage=recyclable_percentage,
        non_recyclable_percentage=non_recyclable_percentage
    )


# Route để xử lý đăng ký người dùng
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=12)

        conn = sqlite3.connect('rubbish_classification.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)',
                           (username, hashed_password, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another one.', 'error')
        finally:
            conn.close()

    return render_template('register.html')


# Route để xử lý đăng nhập người dùng
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('rubbish_classification.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')


# Route để xử lý đăng xuất
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


# Route để xem hồ sơ cá nhân
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please log in to access your profile.', 'error')
        return redirect(url_for('login'))

    return render_template('profile.html', username=session['username'])

# Route để nhận dạng ảnh upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='No file provided')

    file = request.files['file']
    img = load_img(BytesIO(file.read()), target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán phân loại
    prediction = model.predict(img_array)
    class_label = 'Recyclable' if prediction[0] > 0.5 else 'Non-Recyclable'

    # Lưu kết quả vào cơ sở dữ liệu
    log_classification(class_label)

    # Lấy thống kê cập nhật
    stats = get_statistics()
    recyclable_count = stats.get('Recyclable', 0)
    non_recyclable_count = stats.get('Non-Recyclable', 0)
    total = recyclable_count + non_recyclable_count
    recyclable_percentage = (recyclable_count / total * 100) if total > 0 else 0
    non_recyclable_percentage = (non_recyclable_count / total * 100) if total > 0 else 0

    return render_template(
        'index.html',
        result=class_label,
        recyclable_count=recyclable_count,
        non_recyclable_count=non_recyclable_count,
        recyclable_percentage=recyclable_percentage,
        non_recyclable_percentage=non_recyclable_percentage
    )


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

    # Lưu kết quả vào cơ sở dữ liệu
    log_classification(class_label)

    # Lấy thống kê cập nhật
    stats = get_statistics()

    return jsonify({
        'result': class_label,
        'statistics': stats
    })
if __name__ == '__main__':
    app.run(debug=True)
