"""
Fusionne robottest (classes 4) dans RDD_SPLIT (classes 4)
- copie images et labels, renomme pour éviter collision dans RDD_SPLIT
- pas de conversion de classe (normalisation déjà faite)
- vérifie toutes les split train/valid/test
"""

import shutil
from pathlib import Path

RDD_SPLIT = Path(r"C:\Users\4Vents\Desktop\IaRoadDamage\RDD_SPLIT")
ROBOTEST = Path(r"C:\Users\4Vents\Desktop\IaRoadDamage\robottest")

if not RDD_SPLIT.exists():
    raise FileNotFoundError(f"RDD_SPLIT not found: {RDD_SPLIT}")
if not ROBOTEST.exists():
    raise FileNotFoundError(f"robottest not found: {ROBOTEST}")

splits = [("train", "train"), ("valid", "val"), ("test", "test")]

stats = {
    'images_copied': 0,
    'labels_copied': 0,
    'labels_missing': 0,
    'skipped_existing': 0,
}

for src, dst in splits:
    src_images = ROBOTEST / src / "images"
    src_labels = ROBOTEST / src / "labels"
    dst_images = RDD_SPLIT / dst / "images"
    dst_labels = RDD_SPLIT / dst / "labels"

    if not src_images.exists() or not src_labels.exists():
        print(f"⚠️  split robottest inconnu: {src}")
        continue
    if not dst_images.exists() or not dst_labels.exists():
        print(f"⚠️  split RDD inconnu, on crée: {dst}")
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

    dest_existing = set(f.name for f in dst_images.glob("*") if f.is_file())
    idx = len(dest_existing)

    for img in sorted(src_images.glob("*")):
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue

        new_name = f"robottest_{dst}_{idx:06d}{img.suffix.lower()}"
        if new_name in dest_existing:
            stats['skipped_existing'] += 1
            idx += 1
            continue

        dest_img = dst_images / new_name
        shutil.copy2(img, dest_img)
        stats['images_copied'] += 1

        src_lbl = src_labels / f"{img.stem}.txt"
        dst_lbl = dst_labels / f"{Path(new_name).stem}.txt"

        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
            stats['labels_copied'] += 1
        else:
            stats['labels_missing'] += 1

        idx += 1

print("\n=== Fusion robottest -> RDD_SPLIT terminée ===")
print(f"Images copiées: {stats['images_copied']}")
print(f"Labels copiés: {stats['labels_copied']}")
print(f"Labels manquants: {stats['labels_missing']}")
print(f"Skips existants: {stats['skipped_existing']}")
