import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('model_20.h5', compile=False)

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust based on your model input
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Streamlit UI
st.title("Coffee Leaf Disease Detection 🌿")
st.write("Upload an image of a coffee leaf to check if it is infected.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)

    # Adjust labels based on your classes
    classes = ['Healthy', 'Infected']
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence.")
