import os
import hashlib
from collections import defaultdict

def get_file_hash(filepath):
    """Calcule l'empreinte MD5 d'un fichier."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates(root_folder):
    hashes = defaultdict(list)
    total_size_saved = 0
    count = 0

    print(f"--- Analyse de : {root_folder} ---")
    
    for root, _, files in os.walk(root_folder):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                file_hash = get_file_hash(path)
                hashes[file_hash].append(path)
                count += 1
                if count % 1000 == 0:
                    print(f"Fichiers analysés : {count}...")
            except Exception as e:
                print(f"Erreur sur {path}: {e}")

    print("\n--- RÉSULTATS ---")
    duplicates_found = False
    for h, paths in hashes.items():
        if len(paths) > 1:
            duplicates_found = True
            print(f"\n[DOUBLON] {len(paths)} fichiers identiques trouvés :")
            for p in paths:
                size = os.path.getsize(p) / (1024 * 1024)
                print(f"  - {p} ({size:.2f} MB)")
            # Calcul de l'espace gâché (taille du fichier * nombre de copies en trop)
            total_size_saved += os.path.getsize(paths[0]) * (len(paths) - 1)

    if not duplicates_found:
        print("Aucun doublon exact (contenu identique) trouvé.")
    else:
        print(f"\nEspace total perdu en doublons : {total_size_saved / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    target = r"C:\Users\4Vents\Desktop\IaRoadDamage\RDD_SPLIT"
    if os.path.exists(target):
        find_duplicates(target)
    else:
        print("Le chemin n'existe pas.")