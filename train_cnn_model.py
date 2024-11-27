import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# Tạo thư mục logs với timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

# Khai báo TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Đường dẫn đến dữ liệu
data_dir = r'D:\pythonwork\training_project\DATA FOR TRAINING'

# Tạo ImageDataGenerator để đọc dữ liệu đã gán nhãn
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Tạo train_generator và validation_generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),  # Kích thước ảnh
    batch_size=32,
    class_mode='binary',  # Phân loại nhị phân: tái chế hoặc không tái chế
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Kết quả nhị phân
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình với TensorBoard callback
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,
    callbacks=[tensorboard_callback]  # Thêm callback tại đây
)

# Lưu mô hình đã huấn luyện
try:
    model.save(r'D:\pythonwork\training_project\rubbish_classification_model.h5')
    print("Mô hình đã được lưu thành công!")
except Exception as e:
    print("Có lỗi xảy ra khi lưu mô hình:", e)
