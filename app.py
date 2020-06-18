from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import cv2
import numpy as np

# Keras
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = r'C:\Users\hp\Desktop\covid\model\densenet_300.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img/255
    img = np.reshape(img,[1,224,224,3])
    pred = model.predict(img)
    i1 = pred.argmax(axis=-1)
    if(i1==0):
        preds = "This is a Covid-19 Positive Patient"
    else:
        preds = "This is a Covid-19 Negative Patient"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
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

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)