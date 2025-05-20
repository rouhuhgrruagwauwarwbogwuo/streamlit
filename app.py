import streamlit as st
import numpy as np
import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Deepfake Detection - Face Only", layout="centered")

# 初始化 MTCNN 人臉偵測器
detector = MTCNN()

def extract_face(image):
    results = detector.detect_faces(image)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]
        return face
    else:
        return None

def preprocess_face(face_img, target_size=(128, 128)):
    face_resized = cv2.resize(face_img, target_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_norm = face_rgb / 255.0
    return np.expand_dims(face_norm, axis=0).astype(np.float32)

def load_deepfake_model():
    # 這裡改成你自己的模型路徑
    model = load_model("your_deepfake_model.h5")
    return model

def predict_deepfake(model, face_img):
    preprocessed = preprocess_face(face_img)
    pred_prob = model.predict(preprocessed)[0][0]
    label = "Deepfake" if pred_prob > 0.5 else "Real"
    return label, pred_prob

def main():
    st.title("Deepfake 偵測 (只分析臉部)")

    uploaded_file = st.file_uploader("上傳圖片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption="原始圖片", use_container_width=True)

        face = extract_face(img_rgb)
        if face is not None:
            st.image(face, caption="擷取到的人臉", use_container_width=True)

            model = load_deepfake_model()

            label, prob = predict_deepfake(model, face)

            st.markdown(f"### 偵測結果：**{label}**")
            st.markdown(f"### 置信度：{prob:.3f}")

        else:
            st.warning("未偵測到人臉，請換張圖片試試。")

if __name__ == "__main__":
    main()
