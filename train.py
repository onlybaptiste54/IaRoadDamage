import os
import torch
from ultralytics import YOLO

def train():
    # 1. Vérification stricte du chemin YAML
    yaml_path = "/usr/src/app/RDD_SPLIT/data.yaml"
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"ÉCHEC CRITIQUE : Fichier introuvable à l'emplacement {yaml_path}")

    # 2. Vérification Hardware explicite
    if torch.cuda.is_available():
        device = 0
        print(f"Hardware OK : GPU détecté ({torch.cuda.get_device_name(0)})")
    else:
        device = 'cpu'
        print("ALERTE CRITIQUE : CUDA indisponible, exécution forcée sur CPU.")

    # 3. Initialisation du Modèle
    model = YOLO('yolo11m.pt') 

    # 4. Entraînement
    model.train(
        data=yaml_path,
        epochs=20,
        imgsz=512,
        batch=16,
        device=device,
        amp=True,
        project='Aetheria_RDD',
        name='v11_m_rdd_final',
        
        # --- VALIDATION & METRICS ---
        val=True,              
        save=True,             
        plots=True,            
        
        # --- DATA AUGMENTATION ---
        mosaic=1.0,            
        mixup=0.1,             
        hsv_h=0.015,           
        flipud=0.5,            
        
        # --- OPTIMISATION ---
        patience=20,           
        lr0=0.01               
    )

if __name__ == "__main__":
    train()