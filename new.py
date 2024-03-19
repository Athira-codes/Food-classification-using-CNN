import streamlit as st
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tensorflow as tf


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('modelfood.h5')
    return model


def preprocess_image(image):
    image = image.resize((150, 150))  # Resize the image to match the input size of the model
    return image


def predict(image, model):
    preprocessed_image = preprocess_image(image)
    image_array = np.array(preprocessed_image) / 255.0
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    return prediction


def load_image_from_file():
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write("Uploaded Images:")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            model = load_model()
            prediction = predict(image, model)
            display_prediction(prediction)


def load_image_from_url():
    image_url = st.text_input(label='Enter the image URL:')
    if image_url != '':
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Uploaded Image', use_column_width=True)
            model = load_model()
            prediction = predict(image, model)
            display_prediction(prediction)
        except Exception as e:
            st.error("Error loading image from URL. Please check the URL and try again.")


def display_prediction(prediction):
    st.subheader("Prediction:")
    if prediction[0][0] > 0.5:
        st.write(":blue[This is a food image.:heavy_check_mark:]")
    else:
        st.write(":blue[This is not a food image.:x:]")


def main():
    st.title(":green[Food Image Classification]")

    # Choose image source
    image_source = st.radio(":red[Choose image source:]", ("Upload from Computer", "Input Image URL"))

    if image_source == "Upload from Computer":
        load_image_from_file()
    else:
        load_image_from_url()


if __name__ == "__main__":
    main()
