from __future__ import division, print_function
from dotenv import load_dotenv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
load_dotenv()
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = '/home/ubuntu/covid/model/ensemble_3classes.h5'
# MODEL_PATH = r'C:\Users\jaayu\Desktop\projects\covid\model\densenet_300.h5'

MODEL_PATH = os.getenv("MODEL_PATH")

s = time.time()
# Load your trained model
model = load_model(MODEL_PATH)
e = time.time()
print(f'Model loaded! Load Time: {e-s} sec')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img/255
    img = np.reshape(img, [1, 224, 224, 3])
    pred = model.predict((img, img))
    i1 = pred.argmax(axis=-1)
    if i1 == 0:
        preds = "Covid-19 Result: +VE \n\n\t\t|| Confidence: " + str(pred[0][i1]*100).strip("[]") +"%"
    elif i1 == 1:
        preds = "Covid-19 Result: -VE \n\n\t\t|| Confidence: " + str(pred[0][i1]*100).strip("[]") +"%"
    elif i1 == 2:
        preds = "Covid-19 Result: -VE (Pneumonia Detected!) |\n\n\t\t| Confidence: " + str(pred[0][i1]*100).strip("[]") +"%"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    s = time.time()
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print(f)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        os.remove(file_path)

        e = time.time()
        print(f"Prediction Time: {e-s} sec")
        
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=False)
