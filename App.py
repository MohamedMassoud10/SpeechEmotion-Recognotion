import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import keras
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

# Load pre-trained model
model = load_model('model.h5')

# Set page layout
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="wide")

# Title and description
st.title("Speech Emotion Recognition 🎤")
st.write("Upload an audio file to predict the emotion behind the speech.")
st.markdown("This application uses a deep learning model trained on the **TESS** dataset to classify emotions from speech audio. The emotions recognized are **anger**, **fear**, **happy**, **sadness**, **disgust**, **surprise**, and **neutral**.")

# Function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# File upload section
audio_file = st.file_uploader("Upload a .wav or .mp3 file", type=['wav', 'mp3'])

if audio_file is not None:
    # Save the uploaded file to a temporary directory
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Display audio player
    st.audio(audio_file, format='audio/wav')
    
    # Extract MFCC features
    mfcc_features = extract_mfcc("uploaded_audio.wav")
    mfcc_features = np.expand_dims(mfcc_features, -1)  # Add a dummy axis to match input shape

    # Predict emotion
    prediction = model.predict(np.array([mfcc_features]))
    predicted_label = np.argmax(prediction)

    # Map prediction to emotion label
    emotions = ['anger', 'fear', 'happy', 'sadness', 'disgust', 'surprise', 'neutral']
    predicted_emotion = emotions[predicted_label]
    
    # Display prediction result
    st.subheader(f"Predicted Emotion: {predicted_emotion.capitalize()}")
    
    # Display a probability chart for emotions
    st.write("### Prediction Probabilities:")
    emotion_probs = prediction[0]
    prob_df = pd.DataFrame({"Emotion": emotions, "Probability": emotion_probs})
    st.bar_chart(prob_df.set_index('Emotion')['Probability'])

    # Visualization section (Waveform and Spectrogram)
    y, sr = librosa.load("uploaded_audio.wav")
    
    st.write("### Waveform:")
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform of the uploaded speech")
    st.pyplot(plt)
    
    st.write("### Spectrogram:")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title("Spectrogram of the uploaded speech")
    st.pyplot(plt)

# Display training accuracy and loss graphs
if st.button('Show Training Graphs'):
    history = pd.read_csv('history.csv')  # Assuming you saved history during training
    
    st.write("### Training Accuracy and Loss")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].plot(history['epoch'], history['accuracy'], label='Training Accuracy')
    ax[0].plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Accuracy over Epochs')
    ax[0].legend()
    
    ax[1].plot(history['epoch'], history['loss'], label='Training Loss')
    ax[1].plot(history['epoch'], history['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss over Epochs')
    ax[1].legend()
    
    st.pyplot(fig)

