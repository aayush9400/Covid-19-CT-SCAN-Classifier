import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Covid-19 Predictor", layout="centered")

# Load your trained model
MODEL_PATH = os.getenv("MODEL_PATH")
model = load_model(MODEL_PATH)
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Function to predict using the model
def model_predict(img, model):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.asarray(img, dtype=np.float32) / 255
    img = img[:,:,:3]
    img = np.reshape(img, [1, 224, 224, 3])
    pred = model.predict(img)
    i1 = pred.argmax(axis=-1)
    if i1 == 0:
        preds = "Covid Result: +VE "
    else:
        preds = "Covid Result: -VE "
    return preds

st.title("Covid-19 Predictor")
st.markdown("### Upload a CT-Scan image to predict Covid-19 infection status")

st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Choose a CT-Scan image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        if percent_complete == 50:
            image = Image.open(uploaded_file)
        else:
            time.sleep(0.05)
        progress_bar.progress(percent_complete + 1)
    
    progress_bar.empty()  
    st.image(image, caption="Uploaded Image", use_column_width=False)

    if st.button("Predict"):
        with st.spinner('Predicting...'):
            result = model_predict(image, model)
        st.success(result)

# Footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .footer-text {
            position: fixed;
            bottom: 0;
            width: 100%;
            color: gray;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer-text">
        <p>Developed by Aayush Jaiswal</p>
    </div>
    """, unsafe_allow_html=True)
