import streamlit as st
import numpy as np
import os
import cv2
import tempfile
from PIL import Image
import requests
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50, EfficientNetB0, Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖像偵測器（整合 Hugging Face CNN 模型）")

# 下載 Hugging Face 上你的 CNN 模型
@st.cache_resource
def download_model_from_hf(model_url, model_path='deepfake_cnn_model.h5'):
    if not os.path.exists(model_path):
        with st.spinner('下載模型中，請稍候...'):
            r = requests.get(model_url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return model_path

# 載入 ResNet50 / EfficientNet / Xception 預訓練模型 + 你的 CNN 模型
@st.cache_resource
def load_models():
    # 先載入三個大模型
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
    efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
    xception_base = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299,299,3))

    resnet_classifier = Sequential([resnet_base, Dense(1, activation='sigmoid')])
    efficientnet_classifier = Sequential([efficientnet_base, Dense(1, activation='sigmoid')])
    xception_classifier = Sequential([xception_base, Dense(1, activation='sigmoid')])

    # 載入你自己的 CNN 模型
    hf_model_path = download_model_from_hf("https://huggingface.co/wuwuwu123123/deepfake3750/resolve/main/deepfake_cnn_model.h5")
    custom_cnn_model = load_model(hf_model_path)

    return {
        'ResNet50': resnet_classifier,
        'EfficientNet': efficientnet_classifier,
        'Xception': xception_classifier,
        'CustomCNN': custom_cnn_model
    }

# 你之前的圖像處理函式：CLAHE + 銳化 + 高通濾波
def high_pass_filter(img):
    img_np = np.array(img)
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    filtered = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(filtered)

def apply_clahe_sharpen(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(img_clahe, (0,0), 3)
    sharpened = cv2.addWeighted(img_clahe, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

def preprocess_image(img, model_name):
    # 先做你原本的預處理
    img = apply_clahe_sharpen(img)
    img = high_pass_filter(img)

    if model_name == 'Xception':
        img = img.resize((299,299))
        img_array = np.array(img).astype(np.float32)
        return preprocess_xception(img_array)
    elif model_name in ['ResNet50', 'EfficientNet']:
        img = img.resize((224,224))
        img_array = np.array(img).astype(np.float32)
        if model_name == 'ResNet50':
            return preprocess_resnet(img_array)
        else:
            return preprocess_efficientnet(img_array)
    elif model_name == 'CustomCNN':
        # 自訂CNN模型用224x224，像素歸一化即可
        img = img.resize((224,224))
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    else:
        # fallback
        img = img.resize((224,224))
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array

def predict_model(models, img):
    preds = []
    for name, model in models.items():
        x = preprocess_image(img, name)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x, verbose=0)[0][0]
        preds.append(pred)
    return preds

def stacking_predict(models, img, threshold=0.55):
    preds = predict_model(models, img)
    avg = np.mean(preds)
    label = "Deepfake" if avg > threshold else "Real"
    return label, avg

def show_prediction(img, models, threshold=0.55):
    label, confidence = stacking_predict(models, img, threshold)
    st.image(img, caption="輸入圖像", use_container_width=True)
    st.subheader(f"預測結果：**{label}**")
    st.markdown(f"信心分數：**{confidence:.2f}**")

    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], confidence, color='green' if label == "Real" else 'red')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('信心分數')
    st.pyplot(fig)

# 人臉偵測用 OpenCV
def extract_face_opencv(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x,y,w,h = faces[0]
        face = np.array(pil_img)[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

models = load_models()

uploaded_image = st.file_uploader("上傳一張圖像", type=["jpg", "jpeg", "png"])
if uploaded_image:
    pil_img = Image.open(uploaded_image).convert("RGB")
    st.image(pil_img, caption="原始圖像", use_container_width=True)

    face_img = extract_face_opencv(pil_img)
    if face_img:
        st.image(face_img, caption="偵測到人臉", width=300)
        show_prediction(face_img, models)
    else:
        st.info("⚠️ 沒偵測到人臉，使用整張圖像預測")
        show_prediction(pil_img, models)
