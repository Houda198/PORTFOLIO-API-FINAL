import os
import uvicorn
import keras
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import joblib
import pickle
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# --- 1. CHARGEMENT DES MODÈLES ---
# Modèle Cancer (Image)
model_path = os.path.join("models", "model.keras")
model_cancer = keras.models.load_model(model_path)

# Modèle Fraude (XGBoost)
model_fraude_xgb = joblib.load("models/xgboost_best.joblib")

# Modèles Shipping (Basé sur ton dossier v5)
shipping_model = joblib.load("models/best_model_v5.joblib")

shipping_scaler = joblib.load("models/scaler_v5.pkl")

with open("models/best_threshold_v5.pkl", "rb") as f:
    threshold_data = pickle.load(f)
    shipping_threshold = threshold_data['threshold']


# --- 2. INITIALISATION FASTAPI ---
app = FastAPI(title="Backend Portfolio IA - Houda")

# --- 3. MODÈLES DE DONNÉES (Pydantic) ---
class FraudeInput(BaseModel):
    features: List[float]

class ShippingInput(BaseModel):
    # Les features dans le même ordre que lors du fit de ton scaler_v5
    features: List[float]


# --- 4. ROUTES ---
@app.get("/")
def home():
    return {
        "status": "online",
        "owner": "Houda",
        "message": "Bienvenue sur l'API de mon portfolio. Routes: /predict/cancer, /predict/fraude, /predict/shipping"
    }

# --- PROJET 1 : CANCER ---
@app.post("/predict/cancer")
async def predict_cancer(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model_cancer.predict(img_array)
        label = "Malin" if pred[0][0] > 0.5 else "Bénin"
        
        return {"prediction": label, "score": float(pred[0][0])}
    except Exception as e:
        return {"error": str(e)}

# --- PROJET 2 : FRAUDE ---
@app.post("/predict/fraude")
def predict_fraude(data: FraudeInput):
    try:
        input_data = np.array([data.features])
        prediction = model_fraude_xgb.predict(input_data)[0]
        return {
            "prediction_xgboost": int(prediction),
            "status": "FRAUDE DETECTEE" if int(prediction) == 1 else "TRANSACTION NORMALE"
        }
    except Exception as e:
        return {"error": str(e)}

# --- PROJET 3 : SHIPPING ---
@app.post("/predict/shipping")
def predict_shipping(data: ShippingInput):
    try:
        # 1. Formatage de l'input
        input_data = np.array([data.features])
        
        # 2. Application de ton scaler_v5
        input_scaled = shipping_scaler.transform(input_data)
        
        # 3. Prédiction avec gestion du seuil (Threshold)
        if hasattr(shipping_model, "predict_proba"):
            prob = shipping_model.predict_proba(input_scaled)[0][1] # Probabilité de la classe positive
            prediction = 1 if prob >= shipping_threshold else 0
            
            return {
                "prediction": int(prediction),
                "probabilite": float(prob),
                "seuil_applique": float(shipping_threshold),
                "status": "Alerte/Retard" if prediction == 1 else "Normal"
            }
        else:
            # Si c'est une régression (prédiction de prix ou de jours)
            prediction = shipping_model.predict(input_scaled)[0]
            return {
                "prediction": float(prediction),
                "type": "Regression"
            }

    except Exception as e:
        return {"error": str(e)}

# --- 5. LANCEMENT ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)