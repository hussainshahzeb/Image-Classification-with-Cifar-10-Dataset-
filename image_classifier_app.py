import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('cifar10.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title('CIFAR-10 Image Classification')
st.write('Upload an image from the CIFAR-10 dataset to classify.')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Preprocess the image
    img = image.resize((32, 32))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the class of the image
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f'This image is most likely a {predicted_class}.')
