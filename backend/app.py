from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]

# ================== INITIALIZE APP ==================
app = Flask(__name__)
CORS(app)

# ================== LOAD MODELS ==================
# Image encoder (Keras)
image_encoder = load_model('model/image_encoder.h5')

# Scaler for CBC features
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Fusion model (CatBoost)
from catboost import CatBoostClassifier
fusion_model = CatBoostClassifier()
fusion_model.load_model('model/fusion_model.cbm')

# ================== FEATURE ORDER ==================
cbc_features_order = ['Gender', 'Hemoglobin', 'MCV', 'MCH', 'MCHC']

# ================== HELPER FUNCTIONS ==================
def process_cbc(cbc_json):
    """
    Convert frontend CBC JSON to scaled numeric numpy array
    """
    df = pd.DataFrame([cbc_json], columns=cbc_features_order)
    
    # Convert Gender to 0/1 if needed
    df['Gender'] = df['Gender'].map({'Male':0, 'Female':1, 0:0, 1:1})
    
    # Force numeric dtype
    df = df.astype(float)
    
    # Scale
    return scaler.transform(df)

def process_image(img_base64):
    """
    Convert base64 image string to numpy array for encoder
    """
    img_data = base64.b64decode(img_base64.split(',')[-1])
    img = Image.open(BytesIO(img_data)).convert('RGB')
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = image_encoder.predict(img_array)
    return features

# ================== ROUTES ==================
@app.route('/')
def home():
    return "HemoScan backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Get CBC and Image
        cbc_json = data['cbc']
        img_base64 = data['image']
        
        # Process features
        cbc_scaled = process_cbc(cbc_json)
        img_features = process_image(img_base64)
        
        # Fusion: concatenate CBC + image features
        fusion_input = np.hstack((cbc_scaled, img_features))
        
        # Predict
        pred = fusion_model.predict(fusion_input)[0]
        pred_label = 'Anemic' if pred == 1 else 'Non-Anemic'
        
        return jsonify({'prediction': pred_label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# ================== RUN APP ==================
if __name__ == "__main__":
    app.run(debug=True)
