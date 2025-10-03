import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from PIL import Image # NEW: Import Pillow for image processing

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced CNN Fashion Classifier",
    page_icon="🧠",
    layout="wide"
)

# --- Class Names ---
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
MODEL_PATH = 'best_model.keras'

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    """Loads and preprocesses the Fashion-MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat

# --- Model Functions ---
def create_model(learning_rate, dropout_rate):
    """Creates a more advanced CNN model with Batch Normalization."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- NEW: Image Processing Function ---
def process_image(uploaded_file):
    """Converts a user-uploaded image to the model's required format."""
    # Open the image using Pillow
    img = Image.open(uploaded_file)
    
    # Convert to grayscale, resize to 28x28
    img = img.convert('L').resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # IMPORTANT: Invert colors to match MNIST format (white foreground on black background)
    img_array = 255.0 - img_array
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Reshape for the model (1 image, 28x28 pixels, 1 channel)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# --- Sidebar for Hyperparameters ---
st.sidebar.header("⚙️ Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 5, 50, 20)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[1e-2, 1e-3, 1e-4, 1e-5],
    value=1e-3
)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.7, 0.5)

# --- Main App Logic ---
st.title("🧠 Advanced Fashion Item Classifier")
st.write("An enhanced CNN classifier with interactive controls and advanced training techniques.")

x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_data()

model = None

# Load or Train Model
if os.path.exists(MODEL_PATH):
    st.info(f"Loading pre-trained model from `{MODEL_PATH}`...")
    model = load_model(MODEL_PATH)

if st.sidebar.button("Train / Re-train Model"):
    with st.spinner('Training the advanced CNN model... This might take a while.'):
        model = create_model(learning_rate, dropout_rate)
        
        datagen = ImageDataGenerator(
            rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
            zoom_range=0.1, horizontal_flip=True
        )
        train_generator = datagen.flow(x_train, y_train_cat, batch_size=64)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

        history = model.fit(
            train_generator, epochs=epochs, validation_data=(x_test, y_test_cat),
            callbacks=[early_stopping, model_checkpoint], verbose=0
        )
    st.success(f"Model trained successfully and best version saved to `{MODEL_PATH}`!")

if model is None:
    st.warning("No trained model available. Please train a model using the button in the sidebar.")
else:
    # --- NEW FEATURE: USER IMAGE PREDICTION ---
    st.header("👕 Predict Your Own Image")
    uploaded_file = st.file_uploader("Choose an image of a clothing item...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process the image and make a prediction
        with st.spinner('Analyzing image...'):
            processed_img = process_image(uploaded_file)
            prediction = model.predict(processed_img)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction) * 100
        
        with col2:
            st.success(f"Prediction: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.info("Note: The model performs best on images with clear backgrounds, similar to the training data.")

    # --- Model Evaluation and Visualizations ---
    st.header("📊 Model Performance on Test Data")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    st.metric(label="Model Test Accuracy", value=f"{test_acc*100:.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Classification Report", "2. Sample Predictions",
        "3. Confusion Matrix", "4. Sample Dataset Images",
    ])
    
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    with tab1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
        st.text(report)

    with tab2:
        st.subheader("Sample Predictions on Test Images")
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < 16:
                ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
                color = 'green' if y_pred[i] == y_test[i] else 'red'
                ax.set_xlabel(f"Pred: {CLASS_NAMES[y_pred[i]]}\nTrue: {CLASS_NAMES[y_test[i]]}", color=color)
                ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Sample Dataset Images")
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
            ax.set_xlabel(CLASS_NAMES[y_test[i]])
            ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig)