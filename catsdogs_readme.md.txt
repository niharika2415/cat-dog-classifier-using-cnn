🐶🐱 Cat vs Dog Classifier (CNN + Streamlit)

A deep learning project that classifies images as **cat** or **dog** using a Convolutional Neural Network trained on 25,000 images and deployed via **Streamlit**.

# Live App
Click on this link: https://cat-dog-classifier-using-cnn-tfkopfkkc9ibhuaeb72bi7.streamlit.app

# Model
- TensorFlow CNN
- ~80% test accuracy
- Trained in Google Colab (GPU)
- Access the cat_dog _model.h5 on: https://drive.google.com/file/d/1kl-TQMW3Y3M-SBNvOeVABK3JRc2MOQmw/view?usp=sharing

# Files
- `cat_dog_app.py` – Streamlit interface
- `cat_dog_classifier.h5` – Trained model
- `requirements.txt` – Needed packages
- `cat_dog_classifier.ipynb` – Model training notebook

# Run Locally
```bash
pip install -r requirements.txt
streamlit run cat_dog_app.py
