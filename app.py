import os
import base64
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import tensorflow as tf
import gdown
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image

# Set up logging
logging.basicConfig(filename='flask_app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Flask App Configuration
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ASSET_FOLDER = 'assets'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ASSET_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Drive File IDs (Replace with your actual IDs)
file_ids = {
    "denseNet201_epochs_10_batchsize_32_lr_0.0001.bin": "1Dlp1Ez4HUEvwO0EmgbQSVx7hOyxpzbWe",
    # "densenet_epochs_10_batchsize_128_lr_01.1.0001.bin": "1Dlp1Ez4HUEvwO0EmgbQSVx7hOyxpzbWe",
    "vgg.h5": "1Y6qGvT_wmyTHs80JnwKTbUKpGhVJSKHJ",
    # "resnet.h5": "18P1bOVnmsMUNSEUZalE2QGeSfVI8Hveu",
    # "ensemble_epochs_10_batchsize_128_lr_0.1.0001.bin": "18P1bOVnmsMUNSEUZalE2QGeSfVI8Hveu"
}

# Download files from Google Drive
for filename, file_id in file_ids.items():
    file_path = os.path.join(ASSET_FOLDER, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    else:
        print(f"{filename} already exists, skipping.")

# PyTorch Model Definition
class DenseNet201(nn.Module):
    def __init__(self, n_classes):
        super(DenseNet201, self).__init__()
        self.transfer_learning_model = timm.create_model("densenet201", pretrained=True, in_chans=1)

        for param in self.transfer_learning_model.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(1920 * 4 * 4, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.33),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.33),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.transfer_learning_model.forward_features(x)
        x = x.view(-1, 1920 * 4 * 4)
        x = self.classifier(x)
        return x

# Load PyTorch Models
dmodel1_path = os.path.join(ASSET_FOLDER, "denseNet201_epochs_10_batchsize_32_lr_0.0001.bin")
dmodel2_path = os.path.join(ASSET_FOLDER, "denseNet201_epochs_10_batchsize_32_lr_0.0001.bin")

dmodel1 = DenseNet201(3)
dmodel1.load_state_dict(torch.load(dmodel1_path, map_location=torch.device('cpu')))
dmodel1.eval()
#
# dmodel2 = DenseNet201(3)
# dmodel2.load_state_dict(torch.load(dmodel2_path, map_location=torch.device('cpu')))
# dmodel2.eval()

# Load TensorFlow Models
models = {
    # "VGG16": tf.keras.models.load_model(os.path.join(ASSET_FOLDER, "vgg.h5")),
    # "ResNet201": tf.keras.models.load_model(os.path.join(ASSET_FOLDER, "vgg.h5")),
    "DenseNet201 (32 batch)": dmodel1,
    # "DenseNet201 (128 batch)": dmodel2,
    # "Ensemble model": tf.keras.models.load_model(os.path.join(ASSET_FOLDER, "vgg.h5")),
}

logging.info("Models loaded successfully.")

# Class Labels
class_names = ['No', 'Sphere', 'Vortex']

# Image Preprocessing for PyTorch
def preprocess_image_pytorch(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image_tensor

# Image Preprocessing for TensorFlow
def preprocess_image_tf(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Prediction for PyTorch
def predict_pytorch(image_path, model):
    image_tensor = preprocess_image_pytorch(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.cpu().numpy()[0]

# Prediction Function
def get_prediction(image_path, model_name):
    model = models[model_name]
    if "DenseNet201" in model_name:
        predicted_class, score = predict_pytorch(image_path, model)
        predicted_score = score[predicted_class]
    else:
        img_array = preprocess_image_tf(image_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_score = predictions[0][predicted_class]
    return predicted_class, predicted_score

#Flask Routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html') #, models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    model_name = request.form.get('model')
    if model_name not in models:
        return jsonify({"error": "Invalid model selected"}), 400

    predicted_class, predicted_score = get_prediction(file_path, model_name)
    class_name = class_names[predicted_class]

    # Convert image to Base64
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return jsonify({
        "prediction": class_name,
        "score": float(predicted_score),
        "image": f"data:image/png;base64,{encoded_string}"
    })

if __name__ == '__main__':
    app.run(debug=True)
