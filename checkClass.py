import os
from pathlib import Path
from collections import Counter

# ← Chemin Windows local, adapte selon où est ton dataset
labels_dir = Path(r"C:\Users\4Vents\Desktop\IaRoadDamage\RDD_SPLIT\train\labels")

names = ["longitudinal_crack", "transverse_crack",
         "alligator_crack", "other_corruption", "pothole"]

counter = Counter()
files = list(labels_dir.glob("*.txt"))

if not files:
    print(f"ERREUR : Aucun fichier .txt trouvé dans :\n  {labels_dir}")
    print("Vérifie que ce dossier existe et contient bien les labels YOLO.")
else:
    for f in files:
        for line in f.read_text().strip().splitlines():
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1

    total = sum(counter.values())
    print(f"\nDistribution sur {total} annotations ({len(files)} images) :")
    for i, name in enumerate(names):
        n = counter.get(i, 0)
        bar = "█" * int(n / total * 40) if total > 0 else ""
        print(f"  {i} {name:<20} {n:>6} ({n/total*100:.1f}%)  {bar}")