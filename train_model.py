import os
import re
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # pyright: ignore[reportMissingImports]
import pickle

# ================== LOAD CBC DATA ==================
df = pd.read_csv("anemia.csv")
features = ['Gender','Hemoglobin', 'MCV', 'MCH', 'MCHC']
X_cbc = df[features].copy()
y = df['Result'].values   # 1 = Anaemic, 0 = Normal

# Encode Gender if necessary
X_cbc['Gender'] = X_cbc['Gender'].map({'Male':0, 'Female':1})

scaler = StandardScaler()
X_cbc_scaled = scaler.fit_transform(X_cbc)

# ================== AUTOENCODER FOR IMAGES ==================
input_img = Input(shape=(128,128,3))

x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, Flatten()(encoded))
autoencoder.compile(optimizer='adam', loss='mse')

# ================== LOAD AND PROCESS IMAGES ==================
def load_images_from_folder(folder_path):
    images = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.png','.jpg','.jpeg')):
            img = load_img(fpath, target_size=(128,128))
            img = img_to_array(img)/255.0
            images.append(img)
    return np.array(images)

# Patient ID extraction based on filename
def get_patient_id(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

# Directories for images
dataset_folders = ['Conjunctiva', 'Palm', 'Fingernails']

# Map patient IDs to images
patient_image_map = {}  # pid -> list of images
for folder in dataset_folders:
    folder_path = os.path.join(os.getcwd(), folder)
    for subfolder in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(sub_path):
            continue
        for f in os.listdir(sub_path):
            fpath = os.path.join(sub_path, f)
            if os.path.isfile(fpath) and f.lower().endswith(('.png','.jpg','.jpeg')):
                pid = get_patient_id(f)
                if pid is not None:
                    if pid not in patient_image_map:
                        patient_image_map[pid] = []
                    img = load_img(fpath, target_size=(128,128))
                    img = img_to_array(img)/255.0
                    patient_image_map[pid].append(img)

# ================== TRAIN AUTOENCODER ==================
all_images = np.vstack([np.array(imgs) for imgs in patient_image_map.values()])
print(f"Loaded images shape for autoencoder: {all_images.shape}")

autoencoder.fit(all_images, all_images, epochs=10, batch_size=32, verbose=1)

# ================== EXTRACT IMAGE FEATURES ==================
img_features_list = []
for pid in range(1, len(X_cbc_scaled)+1):
    images = patient_image_map.get(pid)
    if images:
        features = encoder.predict(np.array(images))
        features = np.mean(features, axis=0)  # average features per patient
    else:
        features = np.zeros(encoder.output_shape[1])
    img_features_list.append(features)
img_features_fusion = np.array(img_features_list)

# Check shapes
if X_cbc_scaled.shape[0] != img_features_fusion.shape[0]:
    raise ValueError(f"Number of CBC rows ({X_cbc_scaled.shape[0]}) does not match image features ({img_features_fusion.shape[0]})!")

# ================== FUSION ==================
X_fusion = np.hstack((X_cbc_scaled, img_features_fusion))

# ================== TRAIN/TEST SPLIT BY PATIENT ==================
patient_ids = np.arange(X_fusion.shape[0])
train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

X_train, X_test = X_fusion[train_ids], X_fusion[test_ids]
y_train, y_test = y[train_ids], y[test_ids]

fusion_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0)
fusion_model.fit(X_train, y_train)

y_pred = fusion_model.predict(X_test)
print("Fusion shape:", X_fusion.shape)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ================== SAVE MODELS ==================
fusion_model.save_model("fusion_model.cbm")
encoder.save("image_encoder.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Models saved successfully.")
