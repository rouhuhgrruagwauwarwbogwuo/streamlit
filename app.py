import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import requests

# 標題
st.set_page_config(page_title="Deepfake 圖片偵測", layout="centered")
st.title("🕵️‍♂️ Deepfake 圖片偵測系統")

# 模型下載與載入
@st.cache_resource
def load_model():
    model_path = "deepfake_cnn_model.h5"
    hf_url = "https://huggingface.co/wuwuwu123123/deepfake3750/resolve/main/deepfake_cnn_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("正在從 Hugging Face 下載模型..."):
            response = requests.get(hf_url, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ 模型下載完成")
            else:
                st.error("❌ 模型下載失敗")
                st.stop()

    model = tf.keras.models.load_model(model_path)
    return model

# 預處理圖片
def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # 正規化
    img = np.expand_dims(img, axis=0)
    return img

# 主程式邏輯
def main():
    model = load_model()

    uploaded_file = st.file_uploader("請上傳一張圖片（JPG 或 PNG）", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳的圖片", use_column_width=True)

        img_input = preprocess_image(image)
        prediction = model.predict(img_input)[0][0]

        # 顯示預測結果
        label = "🟢 真實影像" if prediction < 0.5 else "🔴 Deepfake 假影像"
        confidence = 1 - prediction if prediction < 0.5 else prediction

        st.markdown(f"### 預測結果：{label}")
        st.progress(float(confidence))
        st.write(f"信心分數：{confidence:.2f}")

if __name__ == "__main__":
    main()
