import os
import sys
from datetime import datetime

import torch
import yaml
from ultralytics import YOLO


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging():
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"train_{timestamp}.log")
    log_file = open(log_path, "a", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    print(f"[LOG] Console tee active -> {log_path}")
    return log_file, original_stdout, original_stderr


def restore_streams(log_file, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()

def train():
    # ✅ Déterminer le répertoire du projet (local ou Docker)
    if os.path.exists("/usr/src/app/RDD_SPLIT/data.yaml"):
        # En Docker
        yaml_path = "/usr/src/app/RDD_SPLIT/data.yaml"
    elif os.path.exists("./RDD_SPLIT/data.yaml"):
        # En local (chemin relatif)
        yaml_path = "./RDD_SPLIT/data.yaml"
    else:
        # Chercher le chemin exact
        raise FileNotFoundError(
            f"ÉCHEC CRITIQUE : data.yaml introuvable.\n"
            f"  Cherché: /usr/src/app/RDD_SPLIT/data.yaml\n"
            f"  Cherché: ./RDD_SPLIT/data.yaml\n"
            f"  CWD: {os.getcwd()}"
        )

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
        epochs=100,             
        imgsz=640,
        rect=False, # Force le padding pour faire un carré parfait de 640x640    
        batch=12,               
        device=device,
        amp=True,
        pretrained=True,
        seed=42,
        project='Aetheria_RDD',
        name='v11_m_rdd_v2_100eRomain', # ✅ Nouveau nom pour ne pas mélanger les logs
        
        # --- AUGMENTATION ---
        mosaic=0.0,
        mixup=0.0,
        flipud=0.0,
        hsv_h=0.005, # Très faible : gère juste les erreurs de balance des blancs
        hsv_s=0.1,   # Modéré : simule l'aspect sec vs aspect mouillé
        hsv_v=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.1,
        erasing=0.1,
        close_mosaic=0, # Désactivé pour éviter les artefacts sur les petites classes

        # --- OPTIMISATION ---
        optimizer='AdamW',
lr0=0.0001,             # ← Ajusté selon votre message précédent        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.01,      
        patience=20,            

        # --- PERFORMANCE ---
        cache=False,            # ← Désactivé pour éviter problèmes torch/dynamo
        workers=8,
    )

    # --- ÉTAPE 2 : ÉVALUATION FINALE SUR TEST SET ---
    print("\n" + "="*70)
    print("🧪 DÉBUT ÉVALUATION SUR TEST SET")
    print("="*70)
    
    # ✅ Récupérer le chemin du meilleur modèle dynamiquement
# ✅ Récupérer le chemin du meilleur modèle dynamiquement
final_weights = os.path.join(results.save_dir, 'weights', 'best.pt')    
    if not os.path.exists(final_weights):
        print(f"❌ ERREUR: Modèle non trouvé à {final_weights}")
        return
        
    try:
        best_model = YOLO(final_weights) # Chargement du dernier poids
        
        # Lancer l'évaluation
   # Lancer l'évaluation
        print(f"⏳ Évaluation en cours sur {yaml_path}...")
        test_results = best_model.val(
            data=yaml_path,
            split='test',
            plots=True,           
            save_json=True,       
            imgsz=640,            # CORRECTION : Doit correspondre à l'entraînement
            batch=12,  
            rect=False,           # ← AJOUT : Stricte parité avec l'entraînement           # CORRECTION : Sécurisation de la VRAM
            device=device
        )
        
        print(f"✅ Évaluation terminée avec succès!")
        
        # --- RÉSULTATS ---
        print(f"\n" + "="*70)
        print(f"📊 RÉSULTATS TEST SET")
        print(f"="*70)
        
        if hasattr(test_results, 'box'):
            print(f"\n🎯 Métriques globales:")
            print(f"   mAP@50:    {test_results.box.map50:.4f}")
            print(f"   mAP@50-95: {test_results.box.map:.4f}")
            
            # Performances par classe
            print(f"\n📊 Performance par classe:")
            names = cfg.get('names', {})
            for i, ap50 in enumerate(test_results.box.ap50):
                class_name = names.get(i, f"Class_{i}") if isinstance(names, dict) else (names[i] if i < len(names) else f"Class_{i}")
                p = test_results.box.p[i]
                r = test_results.box.r[i]
                print(f"   {class_name:<25} | P={p:.3f}  R={r:.3f}  AP@50={ap50:.3f}")
        
        # Localisation des fichiers
        print(f"\n💾 Résultats sauvegardés dans:")
        print(f"   {test_results.save_dir}")
        
        if os.path.exists(test_results.save_dir):
            files = os.listdir(test_results.save_dir)
            print(f"\n   Fichiers générés ({len(files)}):")
            for f in sorted(files)[:15]:  # Afficher les 15 premiers fichiers
                file_path = os.path.join(test_results.save_dir, f)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / 1024
                    print(f"      - {f} ({size:.1f} KB)")
                else:
                    print(f"      - {f}/ (dossier)")
        
        print(f"\n✅ Test set evaluation COMPLETED!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ ERREUR lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    log_file, original_stdout, original_stderr = setup_run_logging()
    try:
        train()
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        raise
    finally:
        restore_streams(log_file, original_stdout, original_stderr)
