import streamlit as st
import numpy as np
import requests
import os
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- 設定頁面 ---
st.set_page_config(page_title="Deepfake 圖片辨識", layout="centered")
st.title("🧠 Deepfake 圖片辨識系統")

# --- 模型下載與載入 ---
@st.cache_resource
def load_cnn_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfake3750/resolve/main/deepfake_cnn_model.h5"
    model_path = "deepfake_cnn_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("🚀 正在從 Hugging Face 下載模型..."):
            r = requests.get(model_url, stream=True)
            if r.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ 模型下載完成！")
            else:
                st.error("❌ 模型下載失敗，請檢查連結或網路狀況。")
                st.stop()

    model = load_model(model_path)
    return model

# --- 預處理圖片 ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 主程式 ---
def main():
    model = load_cnn_model()

    uploaded_file = st.file_uploader("📤 上傳圖片 (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳圖片", use_container_width=True)

        with st.spinner("🧠 預測中..."):
            img_input = preprocess_image(image)
            prediction = model.predict(img_input)[0][0]
            label = "🔴 Deepfake" if prediction >= 0.5 else "🟢 真實 (Real)"
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            confidence = float(confidence)

        st.markdown("### 📢 預測結果")
        st.write(f"模型判斷：**{label}**")
        st.write(f"信心分數：**{confidence:.2%}**")

        # 顯示信心條（轉換為 0~100 的整數）
        st.progress(int(confidence * 100))

if __name__ == "__main__":
    main()
