import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"  

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from PIL import Image
import io

import tensorflow as tf

app = FastAPI(title="Backend Portfolio MALT - IA Multi-Modèles")

model_cancer = tf.keras.models.load_model("models/best_model.keras")
model_fraude_xgb = joblib.load("models/xgboost_best.joblib")


# --- 2. MODÈLE DE DONNÉES (INPUT) ---
class FraudeInput(BaseModel):
    features: List[float] 

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Bienvenue sur l'API de mon portfolio. Routes: /predict/cancer et /predict/fraude"
    }

# --- 3. ROUTE DÉTECTION CANCER ---
@app.post("/predict/cancer")
async def predict_cancer(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        print(f"DEBUG: Image reçue - Taille originale : {image.size}")
        
        image = image.resize((128, 128)) 
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"DEBUG: Shape envoyée au modèle : {img_array.shape}")
        
        pred = model_cancer.predict(img_array)
        label = "Malin" if pred[0][0] > 0.5 else "Bénin"
        
        return {
            "prediction": label,
            "score": float(pred[0][0])
        }
    except Exception as e:
        print(f"ERREUR CANCER : {str(e)}")
        return {"error": str(e), "suggestion": "Vérifie la taille d'entrée."}

# --- 4. ROUTE DÉTECTION FRAUDE ---
@app.post("/predict/fraude")
def predict_fraude(data: FraudeInput):
    try:
        input_data = np.array([data.features])
        print(f"DEBUG: Taille reçue par l'API : {input_data.shape}") 
        
        prediction = model_fraude_xgb.predict(input_data)[0]
        
        return {
            "prediction_xgboost": int(prediction),
            "status": "FRAUDE DETECTEE" if int(prediction) == 1 else "TRANSACTION NORMALE",
            "score_final": "ALERTE" if int(prediction) == 1 else "OK"
        }
    except Exception as e:
        print(f"ERREUR CRITIQUE : {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)