import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Set page config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to check if it's a cat or a dog!")

# Google Drive file ID (replace with yours)
file_id = '1kl-TQMW3Y3M-SBNvOeVABK3JRc2MOQmw' 
url = f'https://drive.google.com/uc?id={file_id}'
model_path = 'cat_dog_model.h5'

# Download model if not already present
if not os.path.exists(model_path):
    with st.spinner('📥 Downloading model... please wait'):
        gdown.download(url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner('🔍 Classifying...'):
            img = preprocess_image(image)
            prediction = model.predict(img)[0][0]

            if prediction > 0.5:
                st.success("🐶 It's a Dog!")
            else:
                st.success("🐱 It's a Cat!")

            st.write(f"**Confidence:** {prediction:.4f}")
