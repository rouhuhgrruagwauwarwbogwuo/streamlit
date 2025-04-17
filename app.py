import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 載入模型
resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_base,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

custom_model = load_model('deepfake_cnn_model.h5')

# 預處理函數
def preprocess_for_models(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img).astype('uint8')

    # For ResNet50
    resnet_input = preprocess_input(np.expand_dims(img_array.copy(), axis=0))

    # For Custom CNN: CLAHE + Normalize
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)

    return resnet_input, custom_input

# 預測函數
def predict(image_path):
    resnet_input, custom_input = preprocess_for_models(image_path)

    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"

    custom_pred = custom_model.predict(custom_input)[0][0]
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"

    return {
        'resnet_label': resnet_label,
        'resnet_confidence': round(resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred, 4),
        'custom_label': custom_label,
        'custom_confidence': round(custom_pred if custom_pred > 0.5 else 1 - custom_pred, 4)
    }

# Streamlit UI
st.title('Deepfake Detection')
st.write('請上傳一張圖片來進行預測')

# 上傳圖片
uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 保存圖片
    img_path = os.path.join("static", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 顯示圖片
    st.image(img_path, caption="上傳的圖片", use_column_width=True)

    # 進行預測
    results = predict(img_path)

    # 顯示預測結果
    st.subheader("預測結果")
    st.write(f"ResNet50 預測: {results['resnet_label']} (信心分數: {results['resnet_confidence']*100:.2f}%)")
    st.write(f"自訂 CNN 預測: {results['custom_label']} (信心分數: {results['custom_confidence']*100:.2f}%)")
