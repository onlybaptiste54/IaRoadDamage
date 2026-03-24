import os
import torch
import yaml
from ultralytics import YOLO

def train():
    yaml_path = "/usr/src/app/RDD_SPLIT/data.yaml"
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"ÉCHEC CRITIQUE : Fichier introuvable à {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = 0
        print(f"Hardware OK : GPU détecté ({torch.cuda.get_device_name(0)})")
    else:
        device = 'cpu'
        print("ALERTE : CPU uniquement.")

    model = YOLO('yolo11m.pt')

    # --- ÉTAPE 1 : ENTRAÎNEMENT ---
    results = model.train(
        data=yaml_path,
        epochs=100,             # ✅ Passage à 100 pour convergence totale
        imgsz=512,              # ✅ Augmentation à 640 pour le mAP@50-95 (précision des boîtes)
        batch=8,               # ✅ On monte à 16 pour stabiliser les gradients (VRAM ~6.5GB)
        device=device,
        amp=True,
        pretrained=True,
        seed=42,
        project='Aetheria_RDD',
        name='v11_m_rdd_v2_100ep', # ✅ Nouveau nom pour ne pas mélanger les logs
        
        # --- AUGMENTATION ---
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        erasing=0.1,
        close_mosaic=10,

        # --- OPTIMISATION ---
        optimizer='AdamW',
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.01,      # ✅ Augmenté légèrement pour prévenir l'overfitting sur 100 epochs
        patience=20,            # ✅ Arrêt auto si ça stagne pendant 20 epochs

        # --- PERFORMANCE ---
        cache=False,            # ← Désactivé pour éviter problèmes torch/dynamo
        workers=8,
    )

    # --- ÉTAPE 2 : ÉVALUATION DYNAMIQUE ---
    print("\n=== Évaluation finale sur test set ===")
    
    # ✅ FIX CRITIQUE : On récupère le chemin dynamiquement depuis le trainer
    # Cela évite l'erreur "FileNotFound" si YOLO ajoute un chiffre au dossier
    best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_weights)
    
    test_results = best_model.val(
        data=yaml_path,
        split='test',
        plots=True,
        save_json=True
    )

    # Breakdown par classe sécurisé
    names = cfg.get('names', {})
    print("\nPerformances par classe (Test Set) :")
    for i, (p, r, ap50) in enumerate(zip(
        test_results.box.p,
        test_results.box.r,
        test_results.box.ap50
    )):
        # Gère si names est une liste ou un dictionnaire
        class_name = names[i] if isinstance(names, list) else names.get(i, f"Class_{i}")
        print(f"  {class_name:<22} P={p:.3f}  R={r:.3f}  AP@50={ap50:.3f}")

if __name__ == "__main__":
    train()