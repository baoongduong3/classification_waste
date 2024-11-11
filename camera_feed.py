import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import sqlite3
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('rubbish_classification_model.h5')
IMG_SIZE = (150, 150)

# Mở camera
cap = cv2.VideoCapture(0)
# Kết nối và tạo bảng
conn = sqlite3.connect('classification_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        classification TEXT
    )
''')
conn.commit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh về kích thước cho mô hình
    img = cv2.resize(frame, IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán phân loại
    prediction = model.predict(img_array)
    class_label = 'Recyclable' if prediction[0] > 0.5 else 'Non-Recyclable'

    # Hiển thị phân loại lên màn hình
    cv2.putText(frame, f'Classification: {class_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Feed', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
