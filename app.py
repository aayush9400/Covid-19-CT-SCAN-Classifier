from __future__ import division, print_function
from dotenv import load_dotenv
# coding=utf-8
import os
import cv2
import numpy as np

# Keras
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
load_dotenv()
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = r'C:\Users\jaayu\OneDrive\Desktop\projects\covid\model\densenet_300.h5'

MODEL_PATH = os.getenv("MODEL_PATH")

# Load your trained model
model = load_model(MODEL_PATH)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('')
# print('Model loaded!')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img/255
    img = np.reshape(img,[1,224,224,3])
    pred = model.predict(img)
    i1 = pred.argmax(axis=-1)
    if(i1==0):
        preds = "Covid Result: +VE "
    else:
        preds = "Covid Result: -VE "
    return preds


@app.route('/', methods=['GET'])
def index():
    # home page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
