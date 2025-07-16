import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("fashion_mnist_model.keras")

# Class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# App title
st.title("👗🥿 Fashion Accessory Classifier")
st.write("Upload a 28x28 grayscale image of a fashion item")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize and preprocess
    image = ImageOps.fit(image, (28, 28), Image.ANTIALIAS)
    img_array = np.asarray(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### 🧠 Predicted Class: **{predicted_class}**")
