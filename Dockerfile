# Utilisation de l'image de base NVIDIA CUDA avec Ubuntu 22.04 et CUDA 12.8
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

FROM tensorflow/tensorflow:2.18.0-gpu


# Configuration pour éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Mise à jour des paquets et installation des dépendances nécessaires
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    nvidia-cuda-toolkit \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# Copier le code de votre application dans le conteneur
COPY . /app

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances Python (si vous avez un fichier requirements.txt)
RUN pip install --upgrade pip && pip install --ignore-installed  -r requirements.txt

# Exposer le port 8080 (ou un autre port selon votre application)
EXPOSE 8080

# Démarrer l'application (ici, un exemple avec Flask)
CMD ["python3", "app.py"]
