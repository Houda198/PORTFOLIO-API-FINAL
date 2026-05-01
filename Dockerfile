# 1. Utiliser une version légère de Python
FROM python:3.9-slim

# 2. Définir le dossier de travail dans le serveur
WORKDIR /app

# 3. Installer les dépendances système (Indispensable pour XGBoost, LightGBM et OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier le fichier requirements.txt
COPY requirements.txt .

# 5. Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout ton code et tes modèles dans le serveur
COPY . .

# 7. Lancer l'API sur le port utilisé par Render
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]