#!/usr/bin/env python3
"""
Évalue le meilleur modèle d'un run YOLO sur le test set
et génère les résultats (plots, json, logs)
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def evaluate_run(run_path: str, data_yaml: str):
    """Évalue un modèle entraîné sur le test set"""
    
    best_weights = os.path.join(run_path, 'weights', 'best.pt')
    
    print(f"\n{'='*60}")
    print(f"📊 ÉVALUATION TEST SET")
    print(f"{'='*60}")
    print(f"📂 Run path: {run_path}")
    print(f"🔍 Weights: {best_weights}")
    print(f"📋 Data config: {data_yaml}")
    
    # Vérifier le modèle existe
    if not os.path.exists(best_weights):
        print(f"❌ ERREUR: Modèle non trouvé à {best_weights}")
        return False
    
    # Charger le modèle
    try:
        model = YOLO(best_weights)
        print(f"✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return False
    
    # Lancer l'évaluation sur le test set
    try:
        print(f"\n⏳ Évaluation en cours sur test set...")
        test_results = model.val(
            data=data_yaml,
            split='test',
            plots=True,           # Génère les courbes
            save_json=True,        # Sauve les résultats en JSON
            imgsz=512,
            batch=16,
            device=0 if __import__('torch').cuda.is_available() else 'cpu'
        )
        
        print(f"✅ Évaluation terminée!")
        
        # Afficher les résultats
        print(f"\n{'='*60}")
        print(f"📈 RÉSULTATS TEST SET")
        print(f"{'='*60}")
        
        # Métriques globales
        if hasattr(test_results, 'box'):
            print(f"\n🎯 Métriques globales:")
            print(f"   mAP@50:    {test_results.box.map50:.4f}")
            print(f"   mAP@50-95: {test_results.box.map:.4f}")
            
            # Par classe
            if hasattr(test_results.box, 'ap50'):
                print(f"\n📊 Performance par classe:")
                with open(data_yaml) as f:
                    cfg = yaml.safe_load(f)
                names = cfg.get('names', {})
                
                for i, ap50 in enumerate(test_results.box.ap50):
                    class_name = names.get(i, f"Class_{i}") if isinstance(names, dict) else names[i]
                    p = test_results.box.p[i]
                    r = test_results.box.r[i]
                    print(f"   {class_name:<25} P={p:.3f}  R={r:.3f}  AP@50={ap50:.3f}")
        
        # Localisation des fichiers générés
        print(f"\n{'='*60}")
        print(f"💾 FICHIERS GÉNÉRÉS")
        print(f"{'='*60}")
        
        val_dir = test_results.save_dir
        print(f"📁 Tous les résultats dans: {val_dir}")
        
        # Lister les fichiers
        if os.path.exists(val_dir):
            print(f"\n   Fichiers créés:")
            for item in os.listdir(val_dir):
                item_path = os.path.join(val_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path) / 1024  # KB
                    print(f"   - {item} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Chemins (à adapter selon votre structure)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    runs_to_evaluate = [
        ("v11_m_rdd_v2_100ep", "RDD_SPLIT"),
    ]
    
    for run_name, data_folder in runs_to_evaluate:
        run_path = os.path.join(BASE_DIR, "runs", "detect", "Aetheria_RDD", run_name)
        data_yaml = os.path.join(BASE_DIR, data_folder, "data.yaml")
        
        if os.path.exists(run_path) and os.path.exists(data_yaml):
            success = evaluate_run(run_path, data_yaml)
            if not success:
                print(f"\n❌ Évaluation échouée pour {run_name}")
        else:
            print(f"❌ Chemin invalide:")
            print(f"   Run: {run_path} (existe: {os.path.exists(run_path)})")
            print(f"   Data: {data_yaml} (existe: {os.path.exists(data_yaml)})")
