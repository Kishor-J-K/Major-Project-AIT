# src/inference.py
import torch
import librosa
import numpy as np
import json

# --- Configuration from your training notebook ---
SR = 22050
DURATION = 3.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
N_SAMPLES = int(SR * DURATION)

# --- Load Labels ---
with open(r"C:\Users\User\OneDrive\Desktop\deploy\audio-classification-webapp\labels.json", "r") as f:
    LABELS = json.load(f)

def preprocess_audio(file_path):
    """
    Converts a single audio file into a Mel spectrogram tensor.
    This function should replicate the preprocessing from your training dataset.
    """
    y, sr = librosa.load(file_path, sr=SR)
    
    # Pad or truncate the audio signal
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
        
    # Create Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    
    # Create a tensor and add the channel dimension
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()
    
    return mel_tensor

def predict(model, audio_tensor):
    """
    Takes a preprocessed tensor and returns the predicted class index.
    """
    with torch.no_grad():
        # Add the batch dimension - THIS IS THE FIX
        # The tensor goes from [channels, height, width] to [batch, channels, height, width]
        audio_tensor = audio_tensor.unsqueeze(0)
        
        # Get model output
        outputs = model(audio_tensor)
        _, predicted_idx = outputs.max(1)
        
    return predicted_idx.item()

def predict_class(file_path, model):
    """
    High-level function to predict the class name from a file path.
    """
    # 1. Preprocess the audio file to get a tensor
    mel_tensor = preprocess_audio(file_path)
    
    # 2. Get the predicted index from the model
    class_idx = predict(model, mel_tensor)
    
    # 3. Return the corresponding class name
    return LABELS[class_idx]