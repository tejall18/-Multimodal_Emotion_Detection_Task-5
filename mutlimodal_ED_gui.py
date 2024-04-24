import tkinter as tk
from tkinter import Label, Button
import cv2
import pyaudio
import wave
import librosa
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the facial expression model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to load the voice tone analysis model
def LoadVoiceToneModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to extract features from an audio chunk
def ExtractAudioFeatures(audio_chunk):
    y, sr = librosa.load(audio_chunk, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to predict voice tone
def PredictVoiceTone(audio_chunk, model):
    feature = ExtractAudioFeatures(audio_chunk)
    prediction = model.predict(np.expand_dims(feature, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

# Initialize Tkinter
top = tk.Tk()
top.geometry('800x600')
top.title('Multimodal Emotion Detection')
top.configure(background='#CDCDCD')

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to detect emotion and voice in real-time
def CaptureVideoAndAudio():
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    facial_model = FacialExpressionModel("model_RTED1.json", "model_weights_RTED1.h5")
    voice_tone_model = LoadVoiceToneModel("model_aVTN.json", "model_weightsVTA.h5")

    label_result.config(text="Capturing video and audio...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # Initialize variables
        x = 0
        y = 0
        w = 0
        h = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Predict emotion
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray[np.newaxis, :, :, np.newaxis] / 255.0
            emotion_pred = EMOTIONS_LIST[np.argmax(facial_model.predict(roi))]

            # Predict voice tone
            data = stream.read(CHUNK)
            frames = [data]

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            predicted_class = PredictVoiceTone(WAVE_OUTPUT_FILENAME, voice_tone_model)
            voice_tone = emotions[predicted_class]

            label_result.config(text="Emotion: {}, Voice Tone: {}".format(emotion_pred, voice_tone))

        cv2.imshow('Multimodal Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    cap.release()
    cv2.destroyAllWindows()

# GUI setup
heading = Label(top, text='Multimodal Emotion Detection', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack(side='top')

upload_button = Button(top, text="Capture Video and Audio", command=CaptureVideoAndAudio, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload_button.pack(side='top', pady=20)

label_result = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label_result.pack(side='bottom', pady=20)

emotions = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

top.mainloop()
