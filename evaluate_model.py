import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Thiết lập đường dẫn tới dữ liệu
data_dir = r'D:\pythonwork\training_project\DATA FOR TRAINING'

# Chuẩn bị bộ sinh dữ liệu cho tập validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),  # Kích thước ảnh đã resize
    batch_size=32,
    class_mode='binary',  # Dữ liệu có 2 lớp
    subset='validation')

# Tải mô hình đã huấn luyện
model = load_model('rubbish_classification_model.h5')

# Đánh giá mô hình trên tập validation
loss, accuracy = model.evaluate(validation_generator)

print("Loss:", loss)
print("Accuracy:", accuracy)
