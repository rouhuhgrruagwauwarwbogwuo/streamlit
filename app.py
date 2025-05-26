import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# 載入模型函數（加入快取避免重複載入）
@st.cache_resource(show_spinner=True)
def load_cnn_model():
    model = load_model('deepfake_cnn_model.h5')
    return model

# 圖片預處理函數
def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    image = image.convert('RGB')  # 確保3通道
    img_array = img_to_array(image) / 255.0  # 標準化
    img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 維度
    return img_array

def main():
    st.title("Deepfake CNN 模型圖片預測")

    model = load_cnn_model()

    uploaded_file = st.file_uploader("上傳圖片 (jpg/png)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='上傳圖片', use_column_width=True)

        img_input = preprocess_image(image)
        prediction = model.predict(img_input)[0][0]  # 取第一張圖、第一個輸出值
        confidence = float(prediction)  # 確保是 float

        # 二分類預測，假設閾值0.5
        label = "Deepfake" if confidence >= 0.5 else "Real"

        st.markdown("### 預測結果")
        st.write(f"模型判斷：**{label}**")
        st.write(f"信心分數：**{confidence:.2%}**")

        # 保證 confidence 在 0~1 之間並轉成 0~100 的整數
        confidence_clamped = max(0.0, min(confidence, 1.0))
        st.progress(int(confidence_clamped * 100))

if __name__ == "__main__":
    main()
