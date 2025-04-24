# Utiliser une image Python légère basée sur Debian
FROM python:3.10-slim

# Éviter les prompts interactifs pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Mise à jour et installation des bibliothèques nécessaires pour les dépendances Python (OpenCV, Flask, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier le contenu de l'application dans le conteneur
COPY . .

# Installer les dépendances Python avec --no-cache-dir pour limiter l’usage mémoire
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par l'application (ex : Flask)
EXPOSE 5000

# Lancer l'application
CMD ["python3", "app.py"]
