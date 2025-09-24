import os
import librosa
import numpy as np
import torch

def load_audio(file_path, sr=22050, duration=3.0):
    y, _ = librosa.load(file_path, sr=sr)
    if len(y) < sr * duration:
        y = np.pad(y, (0, int(sr * duration) - len(y)))
    else:
        y = y[:int(sr * duration)]
    return y

def preprocess_audio(y, n_mels=128, n_fft=1024, hop_length=512):
    mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()
    return mel_tensor