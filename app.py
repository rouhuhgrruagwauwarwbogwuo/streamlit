import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, Concatenate)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ======================== 特徵處理函數 ========================
def resize_and_normalize(img, target_size=(128, 128)):
    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude_spectrum.astype(np.uint8)

def apply_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge = cv2.convertScaleAbs(edge)
    return edge

# ======================== 模型架構 ========================
def conv_branch(input_tensor, l2_reg=1e-4):
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    return x

def build_fusion_model(input_shape=(128, 128, 3)):
    rgb_input = Input(shape=input_shape, name='rgb_input')
    clahe_input = Input(shape=input_shape, name='clahe_input')
    fft_input = Input(shape=(128, 128, 1), name='fft_input')
    edge_input = Input(shape=(128, 128, 1), name='edge_input')

    rgb_branch = conv_branch(rgb_input)
    clahe_branch = conv_branch(clahe_input)
    fft_branch = conv_branch(fft_input)
    edge_branch = conv_branch(edge_input)

    merged = Concatenate()([rgb_branch, clahe_branch, fft_branch, edge_branch])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[rgb_input, clahe_input, fft_input, edge_input], outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================== 資料載入 ========================
def load_dataset(data_dir):
    X_rgb, X_clahe, X_fft, X_edge, y = [], [], [], [], []
    for label_dir, label in [('real', 0), ('fake', 1)]:
        for img_path in glob(os.path.join(data_dir, label_dir, '*')):
            try:
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)

                rgb = resize_and_normalize(img_np)
                clahe = resize_and_normalize(apply_clahe(img_np))
                fft = resize_and_normalize(apply_fft(img_np))
                edge = resize_and_normalize(apply_edge(img_np))

                X_rgb.append(rgb)
                X_clahe.append(clahe)
                X_fft.append(np.expand_dims(fft, -1))
                X_edge.append(np.expand_dims(edge, -1))
                y.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    return np.array(X_rgb), np.array(X_clahe), np.array(X_fft), np.array(X_edge), np.array(y)

# ======================== 主流程 ========================
if __name__ == '__main__':
    data_dir = 'data'  # 檔案結構需為 data/real/*.jpg, data/fake/*.jpg
    X_rgb, X_clahe, X_fft, X_edge, y = load_dataset(data_dir)

    # 使用一個分支的 index 做 train/test split
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

    X_tr = [X_rgb[train_idx], X_clahe[train_idx], X_fft[train_idx], X_edge[train_idx]]
    X_val = [X_rgb[val_idx], X_clahe[val_idx], X_fft[val_idx], X_edge[val_idx]]
    y_tr = y[train_idx]
    y_val = y[val_idx]

    model = build_fusion_model()

    early_stop = EarlyStopping(patience=7, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(factor=0.1, patience=5)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, lr_reduce]
    )

    model.save('deepfake_fusion_model.h5')
