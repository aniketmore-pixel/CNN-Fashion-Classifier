# ==============================================================================
# PART 2: STREAMLIT APP SCRIPT (app.py) FOR INFERENCE
# ==============================================================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="🚀",
    layout="wide"
)

# --- CONSTANTS ---
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
MODEL_PATH = 'best_model.keras'

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_trained_model(path):
    """Loads the pre-trained Keras model from the specified path."""
    if not os.path.exists(path):
        st.error(f"Model file not found at '{path}'. Please ensure the model file is in the same directory as this script.")
        return None
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(uploaded_file):
    """
    Converts a user-uploaded image into the format required by the model.
    1. Opens the image.
    2. Converts to grayscale.
    3. Resizes to 28x28 pixels.
    4. Inverts the colors (to match MNIST's white-on-black format).
    5. Normalizes pixel values to be between 0 and 1.
    6. Reshapes the array for the model's input layer.
    """
    try:
        img = Image.open(uploaded_file).convert('L').resize((28, 28))
        img_array = np.array(img)
        img_array = 255.0 - img_array
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- MAIN APPLICATION ---
st.title("🚀 Fashion Item Classifier")
st.write("Upload an image of a clothing item, and this app will predict its category using a trained Convolutional Neural Network.")
st.info("This app uses a model pre-trained in Google Colab. The code for training can be found separately.")

# Load the model
model = load_trained_model(MODEL_PATH)

if model:
    # --- USER IMAGE PREDICTION SECTION ---
    st.header("👕 Predict Your Own Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner('Analyzing the image...'):
            processed_img = process_image(uploaded_file)
            
            if processed_img is not None:
                prediction = model.predict(processed_img)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100
                
                with col2:
                    st.success(f"Prediction: **{predicted_class_name}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    
                    # Display probabilities
                    st.write("Prediction Probabilities:")
                    prob_data = {CLASS_NAMES[i]: prediction[0][i] for i in range(10)}
                    st.bar_chart(prob_data)
else:
    st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure the model file is present.")