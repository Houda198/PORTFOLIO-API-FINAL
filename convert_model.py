import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")

model = tf.keras.models.load_model("models/best_model.keras")

model.save("model.keras")
print("✅ Modèle resauvegardé avec succès en model.keras !")