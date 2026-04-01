"""
Normalise robottest pour le format RDD_SPLIT/YOLOv11 :
- renomme toutes les images en robottest_<split>_<index>.jpg
- convertit les labels de class 0 -> 4
- crée un rapport
"""

import shutil
from pathlib import Path

ROBOTEST = Path(r"C:\Users\4Vents\Desktop\IaRoadDamage\robottest")
SPLITS = ["train", "valid", "test"]

stats = {
    'images_renamed': 0,
    'labels_updated': 0,
    'labels_missing': 0,
    'errors': []
}

for split in SPLITS:
    src_img_dir = ROBOTEST / split / 'images'
    src_lbl_dir = ROBOTEST / split / 'labels'

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        print(f"⚠️  split manquant: {split} (images ou labels)")
        continue

    files = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    idx = 0

    for img_path in files:
        try:
            new_name = f"robottest_{split}_{idx:06d}.jpg"
            new_img_path = src_img_dir / new_name

            # si image non-JPG, reconvertir
            if img_path.suffix.lower() != '.jpg':
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img.save(new_img_path, 'JPEG', quality=95)
                img_path.unlink()
            else:
                img_path.rename(new_img_path)

            # label
            label_src = src_lbl_dir / f"{img_path.stem}.txt"
            label_dst = src_lbl_dir / f"{new_img_path.stem}.txt"

            if label_src.exists():
                with open(label_src, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]

                converted = []
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    parts[0] = '4'
                    converted.append(' '.join(parts))

                with open(label_dst, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(converted) + ('\n' if converted else ''))
                if label_src != label_dst:
                    label_src.unlink()

                stats['labels_updated'] += 1
            else:
                stats['labels_missing'] += 1

            stats['images_renamed'] += 1
            idx += 1

        except Exception as e:
            stats['errors'].append(f"{img_path}: {e}")

print("\n=== Résumé Normalisation robottest ===")
print(f"Images renommées: {stats['images_renamed']}")
print(f"Labels mis à jour: {stats['labels_updated']}")
print(f"Labels manquants: {stats['labels_missing']}")
if stats['errors']:
    print(f"Erreurs: {len(stats['errors'])}")
    for err in stats['errors'][:10]:
        print(' ', err)
