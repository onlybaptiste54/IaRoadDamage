# Image officielle PyTorch garantie avec CUDA et cuDNN
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /usr/src/app

# CORRECTION : Installation des librairies système graphiques minimales pour OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copie des dépendances
COPY requirements.txt .

# Mise à jour de pip pour éviter le crash de parsing
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Installation de YOLO et outils annexes
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

CMD ["python", "train.py"]