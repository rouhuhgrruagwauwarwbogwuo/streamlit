import os
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 初始化 Flask 應用
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 載入模型
try:
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    resnet_classifier = Sequential([
        resnet_base,
        Dense(1, activation='sigmoid')
    ])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    custom_model = load_model('deepfake_cnn_model.h5')

except Exception as e:
    print(f"Error loading models: {e}")
    raise

# 預處理函數
def preprocess_for_models(image_path):
    try:
        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img).astype('uint8')

        # For ResNet50
        resnet_input = preprocess_input(np.expand_dims(img_array.copy(), axis=0))

        # For Custom CNN: CLAHE + Normalize
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        clahe_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)

        return resnet_input, custom_input

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

# 預測函數
def predict(image_path):
    try:
        resnet_input, custom_input = preprocess_for_models(image_path)

        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"

        custom_pred = custom_model.predict(custom_input)[0][0]
        custom_label = "Deepfake" if custom_pred > 0.5 else "Real"

        return {
            'resnet_label': resnet_label,
            'resnet_confidence': round(resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred, 4),
            'custom_label': custom_label,
            'custom_confidence': round(custom_pred if custom_pred > 0.5 else 1 - custom_pred, 4)
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# 路由處理
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                results = predict(filepath)
                if results is None:
                    return "Error during prediction", 500

                return render_template('result.html', image_path=filepath, **results)

            except Exception as e:
                print(f"Error processing file: {e}")
                return "Error processing file", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
