import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# 載入模型（事先訓練好的全臉模型與局部模型）
full_face_model = load_model('full_face_model.h5')
mouth_model = load_model('mouth_model.h5')
eyes_model = load_model('eyes_model.h5')

detector = MTCNN()

def extract_regions(image):
    # MTCNN 偵測
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None, None, None  # 找不到臉

    face = results[0]
    box = face['box']  # [x, y, w, h]
    keypoints = face['keypoints']

    x, y, w, h = box
    # 裁切整張臉
    face_img = image[y:y+h, x:x+w]

    # 擷取嘴巴區域（根據嘴唇左右、上下擴展區域）
    mouth_center = keypoints['mouth_left'], keypoints['mouth_right']
    mouth_x1 = keypoints['mouth_left'][0] - 10
    mouth_x2 = keypoints['mouth_right'][0] + 10
    mouth_y1 = keypoints['mouth_left'][1] - 10
    mouth_y2 = keypoints['mouth_left'][1] + 20  # 往下多一點
    mouth_img = image[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

    # 擷取眼睛區域（兩眼取中間區塊）
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    eyes_x1 = left_eye[0] - 15
    eyes_x2 = right_eye[0] + 15
    eyes_y1 = left_eye[1] - 15
    eyes_y2 = left_eye[1] + 15
    eyes_img = image[eyes_y1:eyes_y2, eyes_x1:eyes_x2]

    return face_img, mouth_img, eyes_img

def preprocess(img, target_size=(224,224)):
    # 轉RGB，調整大小，正規化等
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # 增加 batch 維度
    return img

def predict(image):
    face_img, mouth_img, eyes_img = extract_regions(image)
    if face_img is None:
        return "No face detected", 0.0

    # 分別預處理
    face_input = preprocess(face_img)
    mouth_input = preprocess(mouth_img)
    eyes_input = preprocess(eyes_img)

    # 各模型預測機率 (假設模型輸出為 [batch, 1] 之概率)
    pred_face = full_face_model.predict(face_input)[0][0]
    pred_mouth = mouth_model.predict(mouth_input)[0][0]
    pred_eyes = eyes_model.predict(eyes_input)[0][0]

    # 加權融合（你可自己調整權重）
    final_score = 0.5 * pred_face + 0.25 * pred_mouth + 0.25 * pred_eyes

    label = "Deepfake" if final_score > 0.5 else "Real"
    return label, final_score

# 測試用
image = cv2.imread('test_image.jpg')
label, score = predict(image)
print(f"判斷結果：{label}，信心分數：{score:.3f}")
