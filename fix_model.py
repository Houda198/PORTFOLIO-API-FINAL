import zipfile
import json
import os
import shutil

def clean_keras_config(obj):
    """Nettoie la config Keras 3.14 pour la rendre compatible Keras 3.3"""
    if isinstance(obj, dict):
        # Supprimer les champs que Keras 3.3 ne connait pas
        obj.pop('quantization_config', None)
        obj.pop('shared_object_id', None)
        
        if obj.get('class_name') == 'DTypePolicy' and 'config' in obj:
            return obj['config'].get('name', 'float32')
        
        # Nettoyer récursivement tout le reste
        return {k: clean_keras_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_keras_config(item) for item in obj]
    else:
        return obj

input_file = "models/best_model.keras"      
output_file = "model_fixed.keras"    

if not os.path.exists(input_file):
    print(f"❌ Fichier non trouvé: {input_file}")
    exit(1)

temp_dir = "temp_keras_fix"

print(f"📦 Extraction de {input_file}...")
with zipfile.ZipFile(input_file, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

print("🔧 Nettoyage de la config JSON...")
config_path = os.path.join(temp_dir, "config.json")
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

config = clean_keras_config(config)

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f)

print("📦 Recompression...")
with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_dir)
            zipf.write(file_path, arcname)

shutil.rmtree(temp_dir)
print(f"✅ Modèle corrigé: {output_file}")

print("🧪 Test de chargement avec TF 2.16...")
import tensorflow as tf
try:
    model = tf.keras.models.load_model(output_file)
    print("✅ CHARGEMENT RÉUSSI !")
    
    model.save("model.keras")
    print("✅ Resauvegardé en 'model.keras' (format compatible Render)")
except Exception as e:
    print(f"❌ Erreur: {e}")