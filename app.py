from flask import Flask, render_template, request, jsonify
import os
import random
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

def get_emotion_code(audio_file_path):
    """
    Analyzes an audio file and returns an emotion code:
    0: anger
    1: happy
    2: neutral
    3: sad
    
    Returns -1 if analysis fails
    """
    try:
        # Load models
        model = load_model("emotion_classifier_rnn.h5", compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
        
        # Process audio
        audio, sr = librosa.load(audio_file_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        mel_spec_resized = cv2.resize(mel_spec_norm, (224, 224))
        mel_spec_rgb = cv2.cvtColor(mel_spec_resized, cv2.COLOR_GRAY2RGB)
        mel_spec_preprocessed = preprocess_input(mel_spec_rgb.astype(np.float32))
        mel_spec_batch = np.expand_dims(mel_spec_preprocessed, axis=0)
        
        # Extract features and predict
        features = feature_extractor.predict(mel_spec_batch, verbose=0)
        prediction = model.predict(features, verbose=0)
        
        return np.argmax(prediction)
        
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        return -1


app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Emotion-to-movie mapping
emotion_movies = {
    1: {"emotion": "Happy", "movies": ["Inside Out", "Zindagi Na Milegi Dobara", "The Pursuit of Happyness", "La La Land", "Up", "The Intouchables", "Amélie", "The Grand Budapest Hotel"]},
    3: {"emotion": "Sad", "movies": ["The Fault in Our Stars", "Schindler's List", "Grave of the Fireflies", "A Beautiful Mind", "The Green Mile", "Hachi: A Dog's Tale", "Manchester by the Sea", "Blue Valentine"]},
    0: {"emotion": "Angry", "movies": ["John Wick", "Mad Max: Fury Road", "Gladiator", "Fight Club", "The Dark Knight", "Kill Bill: Vol. 1", "300", "The Revenant"]},
    2: {"emotion": "Neutral", "movies": ["Forrest Gump", "The Shawshank Redemption", "Good Will Hunting", "Inception", "The Matrix", "Interstellar", "The Godfather", "Pulp Fiction"]},
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-movies', methods=['POST'])
def get_movies():
    data = request.get_json()
    emotion_number = data.get('emotion_number')

    if emotion_number in emotion_movies:
        movies = random.sample(emotion_movies[emotion_number]["movies"], 3)
        response = {
            "emotion": emotion_movies[emotion_number]["emotion"],
            "movies": movies
        }
    else:
        response = {"error": "Invalid emotion number"}
    
    return jsonify(response)

@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if audio_file and allowed_file(audio_file.filename):
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Use the emotion detection model
        emotion_number = get_emotion_code(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        if emotion_number >= 0 and emotion_number in emotion_movies:
            movies = random.sample(emotion_movies[emotion_number]["movies"], 3)
            response = {
                "emotion": emotion_movies[emotion_number]["emotion"],
                "movies": movies
            }
        else:
            response = {"error": "Error detecting emotion"}
        
        return jsonify(response)
    
    return jsonify({"error": "Invalid file type"})

if __name__ == '__main__':
    app.run(debug=False)
    test_audio = "adithya_angry.wav"
    emotion_code = get_emotion_code(test_audio)
    emotions = ["anger", "happy", "neutral", "sad"]
    print(f"Emotion Code: {emotion_code}")
    if emotion_code >= 0:
        print(f"Detected Emotion: {emotions[emotion_code]}")
