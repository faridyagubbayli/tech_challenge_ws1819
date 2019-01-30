import sklearn
import pickle
import subprocess
import os
import librosa
import numpy as np
from sklearn.externals import joblib
import RPi.GPIO as GPIO

# Define Constants
SAMPLE_RATE = 32000
SEGMENT_TIME = 2
RECORD_TIME = 2

# Setup LED Output pin
ledpin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(ledpin, GPIO.OUT, initial=GPIO.LOW)

# Helper functions
def extract_feature(X):
    sample_rate = SAMPLE_RATE
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return np.hstack([mfccs,chroma,mel,contrast,tonnetz])

# Read and process recorded file
def read_process_file():
    signal, fs_rate = librosa.core.load("temp.wav", SAMPLE_RATE)
    N = len(signal)

    duration = int(SEGMENT_TIME * fs_rate)

    signal = signal[: N - (N % duration)]
    samples = np.reshape(signal, (-1, duration))
    features = extract_feature(samples[0, :])

    return np.array(features).astype('float')


# Load Standard Scaler, Gaussian Mixture Model and Isotonic Regression
ss = joblib.load('saved_models/ss.pkl')
gmm_clf = joblib.load('saved_models/gmm_clf.pkl')
isotonic_regressor = joblib.load('saved_models/isotonic_regressor.pkl')
print("Models loaded successfully")

while True: # loop forever
    # Record audio to 'temp.wav'
    subprocess.run(['arecord', '-D', 'hw:1', '-f', 'S16_LE', '-c1', '-r44100', '-d', str(RECORD_TIME), 'temp.wav'])

    # Extract features from recorded audio
    features = read_process_file()
    
    # Scale features in order to normalize and prevent numerical stability issues
    scaled_features = ss.transform(features.reshape(1, -1))


    log_probs = gmm_clf.score_samples(scaled_features)
    probabilities = isotonic_regressor.predict(log_probs)
    predictions = [1 if prob >= 0.5 else 0 for prob in probabilities]

    if predictions[0] == 0: # anomaly detected, turn on LED
        GPIO.output(ledpin, GPIO.HIGH)
    else: # normal workflow, turn off LED
        GPIO.output(ledpin, GPIO.LOW)

GPIO.output(ledpin, GPIO.LOW)
