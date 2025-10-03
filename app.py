import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="CNN Fashion Classifier",
    page_icon="👕",
    layout="wide"
)

# st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Class Names ---
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Model & Data Loading (with caching) ---
@st.cache_resource
def load_data():
    """Loads and preprocesses the Fashion-MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0[..., np.newaxis]
    x_test = x_test.astype('float32') / 255.0[..., np.newaxis]
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat

@st.cache_resource
def create_and_train_model(x_train, y_train_cat):
    """Creates, compiles, and trains the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train_cat, epochs=15, batch_size=64, validation_split=0.1, verbose=0)
    return model, history

# --- Main App Logic ---
st.title("👕 Fashion Item Classifier with CNN")
st.write("A mini-project demonstrating image classification using a Convolutional Neural Network on the Fashion-MNIST dataset.")

# Load data
x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_data()

# Train model (this will only run once due to caching)
with st.spinner('Training the CNN model... This might take a minute.'):
    model, history = create_and_train_model(x_train, y_train_cat)
st.success("Model trained successfully!")

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
st.metric(label="Model Test Accuracy", value=f"{test_acc*100:.2f}%")


# --- Visualizations ---
st.header("📊 Model Performance Visualizations")

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Sample Images",
    "2. Training History",
    "3. Confusion Matrix",
    "4. Test Predictions"
])

with tab1:
    st.subheader("Visualization 1: Sample Dataset Images")
    st.write("A look at 25 random images from the test set.")
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
        ax.set_xlabel(CLASS_NAMES[y_test[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    st.pyplot(fig)

with tab2:
    st.subheader("Visualization 2: Model Training History")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("Visualization 3: Confusion Matrix")
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    st.pyplot(fig)

with tab4:
    st.subheader("Visualization 4: Predictions on Test Images")
    st.write("Green labels are correct predictions, red are incorrect.")
    y_pred_probs = model.predict(x_test) # Re-predict if needed
    y_pred = np.argmax(y_pred_probs, axis=1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < 16: # Display 16 images
            ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
            predicted_label = y_pred[i]
            true_label = y_test[i]
            color = 'green' if predicted_label == true_label else 'red'
            ax.set_xlabel(f"Pred: {CLASS_NAMES[predicted_label]}\nTrue: {CLASS_NAMES[true_label]}", color=color)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    st.pyplot(fig)