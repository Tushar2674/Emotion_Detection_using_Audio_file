import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import librosa
from pydub import AudioSegment
from tensorflow.keras.models import load_model
import sounddevice as sd
from scipy.io.wavfile import write

# Load models
emotion_model = load_model('emotion_model_best.h5')
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

gender_model = load_model('Gender_model.h5')
gender_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

language_model = load_model('Language_model.h5')
language_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = [ 'Angry',  'Calm','Disgust',"Fearful", 'Happy','Neutral', 'Sad', 'Surprised']
gender_labels = ['Female','Male']
language_labels = ['English',"Non-English"]

def convert_to_wav(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = os.path.splitext(file_path)[0] + '.wav'
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        print(f"Error converting {file_path} to WAV: {e}")
        return None

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        features = np.hstack([mfccs_mean, chroma_mean, mel_mean])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def predict_emotion(file_path):
    try:
        if not file_path.lower().endswith('.wav'):
            converted_path = convert_to_wav(file_path)
            if converted_path:
                file_path = converted_path
            else:
                return 'File conversion failed'
        
        features = extract_features(file_path)
        if features is None:
            return 'Feature extraction failed'
        features = features.reshape(1, -1)
        
        gender_pred = gender_model.predict(features)
        print(gender_pred)
        gender = gender_labels[np.argmax(gender_pred)]
        print(gender)
        if gender != 'Female':
            return 'Please upload a female voice'
        
        language_pred = language_model.predict(features)
        print(language_pred)
        language = language_labels[np.argmax(language_pred)]
        print(language)
        if language != 'English':
            return 'Please upload an English voice note'
        
        emotion_pred = emotion_model.predict(features)
        print(emotion_pred)
        emotion = emotion_labels[np.argmax(emotion_pred)]
        print(emotion)
        
        return f'Predicted emotion: {emotion}'
    except Exception as e:
        print("Prediction error:", e)
        return 'Prediction failed'

class EmotionPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Predictor")
        self.root.geometry("400x400")
        
        self.label = tk.Label(root, text="Emotion Predictor", font=("Arial", 18))
        self.label.pack(pady=20)
        
        self.upload_button = tk.Button(root, text="Upload Audio File", command=self.upload_file)
        self.upload_button.pack(pady=10)
        
        self.record_button = tk.Button(root, text="Start Recording", command=self.start_recording)
        self.record_button.pack(pady=10)
        
        self.stop_button = tk.Button(root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)
        
        self.recording = False
        self.fs = 16000  
        self.seconds = 60  

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
        if file_path:
            result = predict_emotion(file_path)
            self.result_label.config(text=result)
    
    def start_recording(self):
        self.recording = True
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.result_label.config(text="Recording...")
        self.audio_data = []
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.fs)
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        wav_path = 'recorded_audio.wav'
        write(wav_path, self.fs, np.array(self.audio_data))
        self.result_label.config(text="Recording stopped. Analyzing...")
        result = predict_emotion(wav_path)
        self.result_label.config(text=result)
    
    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.extend(indata[:, 0])

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionPredictorApp(root)
    root.mainloop()
