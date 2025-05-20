import streamlit as st
import numpy as np
import os
from PIL import Image
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Deepfake 偵測", layout="centered")

@st.cache_resource
def load_custom_cnn_model():
    model_url = "https://huggingface.co/wuwuwu123123/newmodel/resolve/main/deepfake_cnn_model.h5"
    model_path = "deepfake_cnn_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("⬇️ 正在從 Hugging Face 下載自訂模型..."):
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success("✅ 模型下載完成！")

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        model = None
    return model

def preprocess_image(img: Image.Image, target_size=(128, 128)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = img_to_array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return img_array

def main():
    st.title("🧠 Deepfake 圖像偵測系統")
    st.markdown("上傳一張人臉圖片，我們將使用自訂 CNN 模型進行 Deepfake 分析。")

    uploaded_file = st.file_uploader("📷 上傳圖片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳圖片", use_container_width=True)

        model = load_custom_cnn_model()
        if model is None:
            st.stop()

        st.write("模型輸入層 shape:", model.input_shape)

        preprocessed_img = preprocess_image(image, target_size=model.input_shape[1:3])
        st.write("預處理後圖片 shape:", preprocessed_img.shape)

        try:
            prediction = model.predict(preprocessed_img)
            st.write("模型輸出 shape:", prediction.shape)
            prediction_val = prediction[0][0] if prediction.ndim == 2 else prediction[0]
            label = "🔴 假的 Deepfake" if prediction_val > 0.5 else "🟢 真實 Real"
            confidence = prediction_val if prediction_val > 0.5 else 1 - prediction_val

            st.markdown("---")
            st.subheader("🔍 偵測結果")
            st.markdown(f"**判斷：{label}**")
            st.progress(float(confidence))
            st.write(f"信心分數：{confidence:.2%}")
        except ValueError as e:
            st.error(f"模型輸入格式錯誤，請檢查輸入圖片尺寸與格式。錯誤詳情：{e}")
        except Exception as e:
            st.error(f"發生未知錯誤：{e}")

if __name__ == "__main__":
    main()
