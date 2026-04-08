#!/usr/bin/env python3
"""
Fusionne val/ → train/ en préservant la structure YOLO
(images + labels doivent correspondre)
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def get_file_pairs(folder):
    """Retourne {filename_base: (image_path, label_path)}"""
    pairs = defaultdict(dict)
    
    images_dir = os.path.join(folder, 'images')
    labels_dir = os.path.join(folder, 'labels')
    
    if os.path.exists(images_dir):
        for img in os.listdir(images_dir):
            base = os.path.splitext(img)[0]
            pairs[base]['image'] = os.path.join(images_dir, img)
    
    if os.path.exists(labels_dir):
        for lbl in os.listdir(labels_dir):
            base = os.path.splitext(lbl)[0]
            pairs[base]['label'] = os.path.join(labels_dir, lbl)
    
    return pairs

def main():
    base_dir = "RDD_SPLIT"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    print(f"📂 Base: {base_dir}")
    print(f"📂 Train: {train_dir}")
    print(f"📂 Val: {val_dir}")
    
    # Vérifier que les dossiers existent
    if not os.path.exists(val_dir):
        print("❌ Dossier val introuvable!")
        return
    
    # Récupérer les paires
    val_pairs = get_file_pairs(val_dir)
    train_pairs = get_file_pairs(train_dir)
    
    print(f"\n📊 Statistiques actuelles:")
    print(f"   Train: {len(train_pairs)} images+labels")
    print(f"   Val:   {len(val_pairs)} images+labels")
    
    # 🔥 VÉRIFIER LES CONFLITS DE MERGE
    print(f"\n{'='*60}")
    print(f"🔍 VÉRIFICATION DES CONFLITS DE MERGE")
    print(f"{'='*60}")
    
    duplicates = set(train_pairs.keys()) & set(val_pairs.keys())
    
    if duplicates:
        print(f"\n❌ ERREUR: {len(duplicates)} FICHIERS EN DOUBLON (MERGE CONFLICT!):")
        print(f"   Les fichiers suivants existent DÉJÀ dans train et val:\n")
        for dup in sorted(list(duplicates))[:20]:
            print(f"      - {dup}")
        if len(duplicates) > 20:
            print(f"      ... et {len(duplicates) - 20} autres")
        
        print(f"\n   ⚠️  RISK: La fusion écraserait les fichiers train!")
        print(f"   ACTION REQUISE:")
        print(f"      1. Identifier pourquoi il y a des doublons")
        print(f"      2. Nettoyer manually val/ ou train/")
        print(f"      3. Relancer le script\n")
        return False
    else:
        print(f"✅ SAFE: Aucun doublon détecté - fusion sûre!")
        print(f"   Val contient {len(val_pairs)} fichiers uniques")
        print(f"   Aucun conflit avec train/\n")
    
    # Copier les images et labels
    print(f"\n{'='*60}")
    print(f"⏳ FUSION EN COURS")
    print(f"{'='*60}\n")
    
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    
    copied = 0
    errors = []
    
    for base, paths in val_pairs.items():
        try:
            # Image
            if 'image' in paths:
                src = paths['image']
                dest = os.path.join(train_images_dir, os.path.basename(src))
                shutil.copy2(src, dest)
            
            # Label
            if 'label' in paths:
                src = paths['label']
                dest = os.path.join(train_labels_dir, os.path.basename(src))
                shutil.copy2(src, dest)
            
            copied += 1
        except Exception as e:
            errors.append(f"   ❌ {base}: {e}")
    
    # Afficher les résultats
    print(f"✅ {copied} fichiers copiés")
    
    if errors:
        print(f"\n⚠️  Erreurs ({len(errors)}):")
        for err in errors:
            print(err)
    
    # Nouvelles stats
    new_train_pairs = get_file_pairs(train_dir)
    print(f"\n📊 Statistiques après fusion:")
    print(f"   Train: {len(new_train_pairs)} images+labels (avant: {len(train_pairs)})")
    print(f"   Val:   {len(val_pairs)} images+labels à supprimer")
    
    # Proposition suppression val
    print(f"\n💡 PROCHAINES ÉTAPES:")
    print(f"   1. Vérifier les fichiers copiés")
    print(f"   2. Mettre à jour data.yaml (supprimer la ligne 'val')")
    print(f"   3. Supprimer RDD_SPLIT/val/ manuellement")
    print(f"\n   Exemple data.yaml:")
    print(f"      path: ./RDD_SPLIT")
    print(f"      train: train/images")
    print(f"      # val: val/images  ← SUPPRIMER CETTE LIGNE")
    print(f"      test: test/images")
    
if __name__ == "__main__":
    main()
