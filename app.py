import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

st.set_page_config(page_title="Deepfake Detection with FFT Fusion", layout="centered")

def fft_magnitude_spectrum(img_gray):
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    return magnitude_spectrum.astype(np.float32)

def create_cnn_branch(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(inp)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return inp, x

def build_fusion_model(img_shape):
    spatial_inp, spatial_feat = create_cnn_branch(img_shape)
    fft_inp, fft_feat = create_cnn_branch(img_shape[:2] + (1,))
    combined = Concatenate()([spatial_feat, fft_feat])
    output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=[spatial_inp, fft_inp], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(img_bgr, target_size):
    img_bgr = cv2.resize(img_bgr, target_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fft_mag = fft_magnitude_spectrum(img_gray)
    fft_mag = cv2.resize(fft_mag, target_size)
    fft_mag = np.expand_dims(fft_mag, axis=-1)

    return img_rgb.astype(np.float32), fft_mag.astype(np.float32)

def main():
    st.title("Deepfake Detection with Frequency Domain Fusion")

    uploaded_file = st.file_uploader("上傳圖片 (JPG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, caption="圖片", use_container_width=True)


        IMG_SIZE = (128, 128)

        # 建立模型（每次 app 啟動重新建）
        model = build_fusion_model((IMG_SIZE[0], IMG_SIZE[1], 3))

        # 預處理
        spatial_input, fft_input = preprocess_image(img_bgr, IMG_SIZE)
        spatial_input = np.expand_dims(spatial_input, axis=0)
        fft_input = np.expand_dims(fft_input, axis=0)

        # 模型預測 (因沒訓練，預測隨機但可運作)
        pred_prob = model.predict([spatial_input, fft_input])[0][0]

        label = "Deepfake" if pred_prob > 0.5 else "Real"
        st.markdown(f"### 判斷結果：**{label}**")
        st.markdown(f"### 置信度：{pred_prob:.3f}")

if __name__ == "__main__":
    main()
