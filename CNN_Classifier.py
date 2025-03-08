import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, LayerNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

print("\nProgram Starting\n")
# =========================== 1️⃣ SETUP DATA PATH ===========================
dataset_path = "data/"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check the path.")

# =========================== 2️⃣ LABEL MAPPING ===========================
label_mapping = {
    "anger": 0, 
    "happy": 1, 
    "neutral": 2, 
    "sad": 3
}

# =========================== 3️⃣ ADVANCED DATA AUGMENTATION ===========================
def augment_audio(audio, sr):
    # Time stretching
    stretched = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    
    # Pitch shifting
    pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 3))
    
    # Add white noise
    noise = np.random.normal(0, 0.005, len(audio))
    noisy = audio + noise
    
    return [audio, stretched, pitched, noisy]

# =========================== 4️⃣ FEATURE EXTRACTION USING EFFICIENTNET ===========================
def audio_to_features(file_path):
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

# =========================== 5️⃣ LOAD DATASET ===========================
X, Y = [], []

# Iterate through speaker folders (1-8)
for speaker_folder in os.listdir(dataset_path):
    if not speaker_folder.isdigit() or int(speaker_folder) < 1 or int(speaker_folder) > 8:
        continue
    
    speaker_path = os.path.join(dataset_path, speaker_folder)
    
    # Iterate through session folders 
    for session_folder in os.listdir(speaker_path):
        session_path = os.path.join(speaker_path, session_folder)
        
        # Iterate through emotion folders
        for emotion_folder in os.listdir(session_path):
            emotion_path = os.path.join(session_path, emotion_folder)
            
            if emotion_folder.lower() in label_mapping and os.path.isdir(emotion_path):
                for file in os.listdir(emotion_path):
                    if file.endswith(".wav"):
                        file_path = os.path.join(emotion_path, file)
                        label = label_mapping[emotion_folder.lower()]
                        feature = audio_to_features(file_path)

                        X.append(feature)
                        Y.append(label)

# Print total number of samples
print(f"Total samples loaded: {len(X)}")

# Ensure we have data
if not X:
    raise ValueError("No audio files found in the dataset. Please check your data structure.")

# Convert features to numpy array
X = np.vstack(X)
Y = np.array(Y)

print(f"Feature shape: {X.shape}")

# One-hot encode labels
Y = tf.keras.utils.to_categorical(Y, num_classes=len(label_mapping))

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# =========================== 6️⃣ FEATURE EXTRACTION ===========================
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Extract features
X_train_features = feature_extractor.predict(X_train, verbose=1)
X_test_features = feature_extractor.predict(X_test, verbose=1)

print(f"Training features shape: {X_train_features.shape}")
print(f"Testing features shape: {X_test_features.shape}")

# =========================== 7️⃣ CLASS WEIGHT HANDLING ===========================
class_weights = compute_class_weight('balanced', classes=np.arange(len(label_mapping)), y=np.argmax(Y, axis=1))
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# =========================== 8️⃣ AGENT AI TECHNIQUE: TRANSFORMER-BASED CLASSIFIER ===========================
classifier = Sequential([
    Dense(512, activation='relu', input_shape=(1280,)),  # Changed input shape to match EfficientNetB0
    LayerNormalization(),
    Dropout(0.4),
    
    Dense(256, activation='relu'),
    LayerNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    LayerNormalization(),
    Dropout(0.2),
    
    Dense(len(label_mapping), activation='softmax')
])

# Compile Model using AdamW (Optimized Adam)
classifier.compile(optimizer=AdamW(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduler & Early Stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train Classifier
history = classifier.fit(
    X_train_features, Y_train, 
    validation_data=(X_test_features, Y_test), 
    epochs=200, 
    batch_size=32, 
    class_weight=class_weights_dict,
    callbacks=[lr_scheduler, early_stopping]
)

# Evaluate Model
loss, acc = classifier.evaluate(X_test_features, Y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# =========================== 9️⃣ PERFORMANCE ANALYSIS ===========================
Y_test_labels = np.argmax(Y_test, axis=1)
Y_pred_probs = classifier.predict(X_test_features)
Y_pred_labels = np.argmax(Y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(Y_test_labels, Y_pred_labels, target_names=label_mapping.keys()))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(Y_test_labels, Y_pred_labels), annot=True, cmap="Blues", fmt="d", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Save Model
classifier.save("emotion_classifier_rnn.h5")
print("Model saved as emotion_classifier_rnn.h5")