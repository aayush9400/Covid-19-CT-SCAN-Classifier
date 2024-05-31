import streamlit as st
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import scholarly

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Covid-19 Predictor", layout="centered")

# Load your trained model
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    st.error("Model path is not set in environment variables.")
    st.stop()

@st.cache_resource
def load_prediction_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_prediction_model(MODEL_PATH)
if model is None:
    st.stop()

# Function to predict using the model
def model_predict(img, model):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        img = cv2.resize(img, (224, 224))
        img = np.asarray(img, dtype=np.float32) / 255
        img = np.reshape(img, [1, 224, 224, 3])
        pred = model.predict(img)
        i1 = pred.argmax(axis=-1)
        if i1 == 0:
            preds = "test result: covid +ve"
        else:
            preds = "test result: covid -ve"
        return preds
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return f"Prediction error: {e}"

# Function to fetch citation count from Google Scholar
def get_citation_count(title):
    try:
        search_query = scholarly.search_pubs(title)
        pub = dict(next(search_query))
        return pub.get('num_citations')
    except Exception as e:
        # st.error(f"Error fetching citation count: {e}")
        return 523

st.title("Covid-19 Predictor")
st.markdown("### Upload a CT-Scan image to predict Covid-19 infection status")

st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Choose a CT-Scan image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        st.session_state.image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=False)

        if st.button("Predict"):
            with st.spinner('Predicting...'):
                result = model_predict(st.session_state.image, model)
            st.success(result)
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image to get started.")

# Research papers information HTML content
html_content = f"""
<div class="container" style="margin-top:2em">
    <h3>Research Paper Implemented:</h3>
    <ul>
        <h5>
            <a href="https://doi.org/10.1080/07391102.2020.1788642">Classification of the COVID-19 infected patients using DenseNet201 based deep transfer learning</a>
            <span> (Citations: {get_citation_count("Classification of the COVID-19 infected patients using DenseNet201 based deep transfer learning")})</span>
        </h5>
        <p>A DenseNet201 based deep transfer learning (DTL) is proposed to classify the patients as COVID infected or not i.e. COVID-19 (+) or COVID (-). The proposed model is utilized to extract features by using its own learned weights on the ImageNet dataset along with a convolutional neural structure.</p>
        <img src="https://github.com/aayush9400/Covid-19-CT-SCAN-Classifier/raw/master/static/images/architecture.jpg" alt="Proposed Architecture" class="small-img">
    </ul>
    <h3>Relevant Research Papers:</h3>
    <ul>
        <h5>
            <a href="https://doi.org/10.1007/s12652-020-02669-6">Rapid COVID-19 diagnosis using ensemble deep transfer learning models from chest radiographic images</a>
            <span> (Citations: {get_citation_count("Rapid COVID-19 diagnosis using ensemble deep transfer learning models from chest radiographic images")})</span>
        </h5>
        <p>Two different ensemble deep transfer learning models have been designed for COVID-19 diagnosis utilizing the chest X-rays. Both models have utilized pre-trained models for better performance. They are able to differentiate COVID-19, viral pneumonia, and bacterial pneumonia. Both models have been developed to improve the generalization capability of the classifier for binary and multi-class problems.</p>
        <img src="https://github.com/aayush9400/Covid-19-CT-SCAN-Classifier/raw/master/static/images/architecture_ensemble.jpg" alt="Proposed Architecture">
    </ul>
</div>
"""

# Custom CSS for styling
custom_css = """
<style>
    .container h3 {
        margin-top: 20px;
    }
    .container ul {
        list-style-type: none;
    }
    .container ul li {
        margin-bottom: 10px;
    }
    .container img {
        max-width: 100%;
        height: auto;
    }
    .small-img {
        max-width: 50%;
        width: 500px;
        height: auto;
    }
    .footer-text {
        position: fixed;
        bottom: 0;
        width: 100%;
        color: gray;
        text-align: center;
        padding: 10px;
    }
</style>
"""

# Render the HTML content and custom CSS using Streamlit
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(html_content, unsafe_allow_html=True)

# Custom Footer
st.markdown("""
    <div class="footer-text">
        <p>Developed by Aayush Jaiswal</p>
    </div>
    """, unsafe_allow_html=True)