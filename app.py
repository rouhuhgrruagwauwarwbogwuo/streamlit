import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

st.set_page_config(page_title="Deepfake Detection with HPF + FFT Fusion", layout="centered")

def high_pass_filter(img_gray):
    # 使用簡單的高通濾波器（Laplacian）
    hpf = cv2.Laplacian(img_gray, cv2.CV_32F)
    hpf = cv2.normalize(hpf, None, 0, 1, cv2.NORM_MINMAX)
    return hpf.astype(np.float32)

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
    # 空間分支 (原圖RGB)
    spatial_inp, spatial_feat = create_cnn_branch(img_shape)

    # 頻域分支 (高通+FFT)
    freq_inp, freq_feat = create_cnn_branch(img_shape[:2] + (1,))  # 單通道頻域圖

    combined = Concatenate()([spatial_feat, freq_feat])
    output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=[spatial_inp, freq_inp], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(img_bgr, target_size):
    img_bgr = cv2.resize(img_bgr, target_size)

    # 空間分支輸入
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0

    # 灰階圖
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 頻域特徵：高通濾波器 + FFT magnitude
    hpf_img = high_pass_filter(img_gray)
    fft_img = fft_magnitude_spectrum(img_gray)
    freq_img = hpf_img * fft_img  # element-wise 乘法加強高頻特徵
    freq_img = cv2.resize(freq_img, target_size)
    freq_img = np.expand_dims(freq_img, axis=-1)

    return img_rgb.astype(np.float32), freq_img.astype(np.float32)

def main():
    st.title("Deepfake Detection with HPF + FFT Fusion")

    uploaded_file = st.file_uploader("上傳圖片 (JPG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(img_bgr, caption="原始圖片", use_container_width=True)

        IMG_SIZE = (128, 128)

        model = build_fusion_model((IMG_SIZE[0], IMG_SIZE[1], 3))

        spatial_input, freq_input = preprocess_image(img_bgr, IMG_SIZE)
        spatial_input = np.expand_dims(spatial_input, axis=0)
        freq_input = np.expand_dims(freq_input, axis=0)

        pred_prob = model.predict([spatial_input, freq_input])[0][0]
        label = "Deepfake" if pred_prob > 0.5 else "Real"
        st.markdown(f"### 判斷結果：**{label}**")
        st.markdown(f"### 置信度：{pred_prob:.3f}")

if __name__ == "__main__":
    main()
