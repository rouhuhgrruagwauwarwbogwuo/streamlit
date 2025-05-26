import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
import cv2
from PIL import Image

# ------------------------
# 模型下載與載入
# ------------------------
@st.cache_resource
def load_model():
    model_path = "deepfake_cnn_model.h5"
    hf_url = "https://huggingface.co/wuwuwu123123/deepfake3750/resolve/main/deepfake_cnn_model.h5"

    # 如果模型不存在，就從 Hugging Face 下載
    if not os.path.exists(model_path):
        with st.spinner("正在從 Hugging Face 下載模型..."):
            response = requests.get(hf_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success("模型下載完成")

    # 載入模型
    model = tf.keras.models.load_model(model_path)
    return model

# ------------------------
# 圖像預處理函式
# ------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0  # 根據你的模型調整大小
    img_input = np.expand_dims(img_resized, axis=0)
    return img_input

# ------------------------
# 主程式
# ------------------------
def main():
    st.set_page_config(page_title="Deepfake 圖像偵測", layout="centered")
    st.title("🧠 Deepfake 圖像偵測系統")
    st.write("上傳圖片，系統將使用自訂 CNN 模型進行真實與偽造的辨識。")

    model = load_model()

    uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳的圖片", use_container_width=True)

        # 預處理圖片
        img_input = preprocess_image(image)

        # 模型預測
        prediction = model.predict(img_input)[0][0]
        label = "🟢 真實影像 (Real)" if prediction < 0.5 else "🔴 偽造影像 (Deepfake)"

        st.subheader("預測結果")
        st.write(f"模型判斷：**{label}**")
        st.progress(float(prediction) if prediction > 0.5 else 1 - float(prediction))

        st.write(f"信心分數：`{prediction:.4f}`（接近 1 表示 Deepfake，接近 0 表示 Real）")

if __name__ == "__main__":
    main()
