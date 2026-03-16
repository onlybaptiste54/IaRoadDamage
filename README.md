# Plan — Réentraînement YOLO11m RDD2022

## Config finale
- **Modèle** : YOLO11m (yolo11m.pt)
- **imgsz** : 512 (résolution native des images Kaggle)
- **batch** : 32 · **epochs** : 100 · **workers** : 12
- **Preprocessing** : CLAHE sur toutes les images (rehausse contraste des cracks)
- **Augmentations** : mosaic=1.0, copy_paste=0.3, mixup=0.1, degrees=5
- **Hardware** : RTX 2080 Super + i9 14900KF · ~5-8h

---

## Architecture des fichiers

```
4VentsIa/
├── data/
│   └── RDD2022/                    ← dataset Kaggle décompressé ici
│       ├── Japan/train/{images/, annotations/xmls/}
│       ├── India/train/...
│       ├── Czech/train/...
│       ├── Norway/train/...
│       ├── United_States/train/...
│       ├── China_MotorBike/train/...
│       └── China_Drone/            ← IGNORÉ (aérien)
│
├── dataset/rdd2022_full/           ← généré par 0_prepare_dataset.py
│   ├── images/{train/, val/}
│   ├── labels/{train/, val/}
│   └── rdd2022.yaml
│
├── runs/train/rdd2022_yolo11m/     ← généré par 1_train.py
│   └── weights/best.pt             ← TARGET
│
├── 0_prepare_dataset.py            ← XML→YOLO + CLAHE + split 80/20
├── 1_train.py                      ← entraînement YOLO11m
├── 2_evaluate.py                   ← comparaison old vs new + inférence vidéo
└── requirements_train.txt          ← dépendances venv natif
```

---

## Fichiers à créer

### `requirements_train.txt`
```
torch==2.5.1+cu124
torchvision==0.20.1+cu124
--extra-index-url https://download.pytorch.org/whl/cu124
ultralytics>=8.3.0
opencv-python
matplotlib
pandas
tqdm
pillow
```

### `0_prepare_dataset.py`
- Parse XML PascalVOC → format YOLO .txt (classes D00/D10/D20/D40 seulement)
- Applique CLAHE (clipLimit=2.0, tileGridSize=8x8) sur chaque image
- Split stratifié par pays 80/20
- Préfixe noms fichiers par pays (évite collisions Japan/India)
- Écrit `rdd2022.yaml`

### `1_train.py`
- Vérifie GPU + CUDA
- Lance `YOLO("yolo11m.pt").train(data=..., imgsz=512, batch=32, epochs=100, ...)`
- Sauvegarde checkpoint tous les 10 epochs

### `2_evaluate.py`
- Val sur les deux modèles (YOLO11m new vs YOLOv8s old)
- Tableau comparaison mAP50 / mAP50-95 / AP50 par classe
- Inférence visuelle sur `road_san_diego_1080p60.mp4`
- Déploie `best.pt` → `roadai/backend/models/best.pt`

---

## Fichier critique à mettre à jour après déploiement
- [requirements-gpu.txt](roadai/backend/requirements-gpu.txt) : bumper `ultralytics==8.2.18` → `ultralytics>=8.3.0`
