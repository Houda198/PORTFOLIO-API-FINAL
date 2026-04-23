import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# --- 1. CHARGEMENT DES MODÈLES ---
model_path = os.path.join("models", "model.keras")
model_cancer = tf.keras.models.load_model(model_path)

model_fraude_xgb = joblib.load("models/xgboost_best.joblib")

# --- 2. INITIALISATION FASTAPI ---
app = FastAPI(title="Backend Portfolio MALT - IA Multi-Modèles")

# --- 3. MODÈLE DE DONNÉES ---
class FraudeInput(BaseModel):
    features: List[float]

# --- 4. ROUTES ---
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Bienvenue sur l'API de mon portfolio. Routes: /predict/cancer et /predict/fraude"
    }

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
        
        return {
            "prediction": label,
            "score": float(pred[0][0])
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/fraude")
def predict_fraude(data: FraudeInput):
    try:
        input_data = np.array([data.features])
        prediction = model_fraude_xgb.predict(input_data)[0]
        
        return {
            "prediction_xgboost": int(prediction),
            "status": "FRAUDE DETECTEE" if int(prediction) == 1 else "TRANSACTION NORMALE",
            "score_final": "ALERTE" if int(prediction) == 1 else "OK"
        }
    except Exception as e:
        return {"error": str(e)}

# --- 5. LANCEMENT ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)