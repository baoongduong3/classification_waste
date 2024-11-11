import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Thiết lập đường dẫn tới dữ liệu
data_dir = r'D:\pythonwork\training_project\DATA FOR TRAINING'

# Chuẩn bị bộ sinh dữ liệu và gán nhãn
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Chia thành tập train và validation

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),  # Kích thước ảnh đã resize
    batch_size=32,
    class_mode='binary',  # Dữ liệu có 2 lớp
    subset='training')

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation')