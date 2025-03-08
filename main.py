import numpy as np
import librosa
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# =========================== 1️⃣ LOAD THE TRAINED MODEL ===========================
model = load_model("emotion_classifier_rnn.h5", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded successfully.")

# EfficientNetB0 Feature Extractor
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Label Mapping
label_mapping = ["anger", "happy", "neutral", "sad"]

# =========================== 2️⃣ FEATURE EXTRACTION FUNCTION ===========================
def audio_to_melspectrogram(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 255]
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        
        # Resize and convert to RGB
        mel_spec_resized = cv2.resize(mel_spec_norm, (224, 224))
        mel_spec_rgb = cv2.cvtColor(mel_spec_resized, cv2.COLOR_GRAY2RGB)
        
        # Preprocess for EfficientNetB0
        mel_spec_preprocessed = preprocess_input(mel_spec_rgb.astype(np.float32))
        return np.expand_dims(mel_spec_preprocessed, axis=0)
    
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None

# =========================== 3️⃣ EMOTION PREDICTION FUNCTION ===========================
def predict_emotion(audio_file):
    # Get mel spectrogram
    mel_spec = audio_to_melspectrogram(audio_file)
    if mel_spec is None:
        return
    
    # Extract features using EfficientNetB0
    features = feature_extractor.predict(mel_spec, verbose=0)
    
    # Make prediction
    prediction = model.predict(features, verbose=0)
    
    predicted_index = np.argmax(prediction)
    predicted_label = label_mapping[predicted_index]
    confidence = np.max(prediction) * 100

    print(f"Predicted Emotion: {predicted_label.capitalize()} ({confidence:.2f}%)")
    print("\nFull Prediction Probabilities:")
    for label, prob in zip(label_mapping, prediction[0]):
        print(f"{label.capitalize()}: {prob*100:.2f}%")

# =========================== 4️⃣ TEST THE MODEL ===========================
if __name__ == "__main__":
    test_audio = "adithya_angry.wav"
    print(f"\nTesting {test_audio}...")
    predict_emotion(test_audio)