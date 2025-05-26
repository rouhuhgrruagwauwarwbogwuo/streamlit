import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# --- 網頁設定 ---
st.set_page_config(page_title="Deepfake 圖片辨識系統", layout="centered")
st.title("🧠 Deepfake 圖片辨識系統")
st.write("請上傳一張圖片，系統將判斷其真偽。")

# --- 載入訓練好的 CNN 模型 ---
@st.cache_resource
def load_cnn_model():
    model = load_model("deepfake_cnn_model.h5")
    return model

model = load_cnn_model()

# --- 圖片預處理函數 ---
def preprocess_image(image: Image.Image):
    # 轉換為 numpy 陣列並確保為 RGB
    img = np.array(image.convert("RGB"))
    
    # 調整尺寸為模型輸入大小
    img = cv2.resize(img, (256, 256))
    
    # 正規化像素值到 [0, 1]
    img = img.astype("float32") / 255.0

    # 增加 batch 維度 -> (1, 256, 256, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

# --- 使用者上傳圖片 ---
uploaded_file = st.file_uploader("請上傳一張圖片 (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的圖片", use_container_width=True)

    # --- 模型預測 ---
    with st.spinner("🔍 分析中，請稍候..."):
        preprocessed_img = preprocess_image(image)
        prediction = model.predict(preprocessed_img)[0][0]

        # --- 判斷結果 ---
        if prediction < 0.5:
            label = "🟢 真實 (Real)"
            confidence = 1 - prediction
        else:
            label = "🔴 假的 (Deepfake)"
            confidence = prediction

        # --- 顯示結果 ---
        st.markdown("### 📢 預測結果")
        st.write(f"模型判斷：**{label}**")
        st.write(f"信心分數：**{confidence:.2%}**")

        # --- 信心分數條 ---
        st.progress(min(confidence, 1.0))
