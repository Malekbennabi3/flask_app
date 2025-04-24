# Image TensorFlow CPU only
FROM tensorflow/tensorflow:2.18.0

# Eviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Mise à jour et installation des dépendances utiles
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# Copier le code dans le conteneur
COPY . /app

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances Python
RUN pip install --upgrade pip && pip install --ignore-installed -r requirements.txt

# Exposer le port (ajustez selon votre app)
EXPOSE 5000

# Commande pour démarrer l'app
CMD ["python3", "app.py"]
