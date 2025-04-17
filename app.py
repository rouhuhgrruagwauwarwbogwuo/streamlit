import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import requests
from tensorflow.keras.models import load_model

# 設定 Hugging Face 模型網址（換成你的連結）
MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfake/blob/main/deepfake_cnn_model.h5"

# 嘗試下載模型到暫存資料夾
@st.cache_resource
def download_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return load_model(model_path)

model = download_model()

# 圖片預處理
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Resize to model input size
    
    # CLAHE 灰階增強
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    img = img / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)

# Streamlit UI
st.title("🕵️ Deepfake 偵測 App")

uploaded_file = st.file_uploader("上傳一張圖片", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # Read the image

    st.image(img, caption="你上傳的圖片", use_column_width=True)

    processed = preprocess_image(img)
    prediction = model.predict(processed)[0][0]
    label = "Deepfake" if prediction > 0.5 else "Real"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🔍 預測結果: **{label}**")
    st.markdown(f"### 📊 信心分數: **{confidence:.2%}**")
