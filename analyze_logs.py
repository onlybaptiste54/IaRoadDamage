import re

with open('docker_logs.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Extraire les dernières lignes de chaque epoch
epochs_data = {}
for epoch in range(1, 16):
    pattern = rf'\s+{epoch}/20\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    matches = list(re.finditer(pattern, content))
    
    if matches:
        last_match = matches[-1]
        gpu_mem, box_loss, cls_loss, dfl_loss = last_match.groups()
        epochs_data[epoch] = {
            'gpu': float(gpu_mem),
            'box_loss': float(box_loss),
            'cls_loss': float(cls_loss),
            'dfl_loss': float(dfl_loss)
        }

# Afficher
print('=' * 80)
print('RESULTAT FINAL - 15 EPOCHS D\'ENTRAINEMENT')
print('=' * 80)
print()
print('| Epoch | GPU(GB) | Box Loss | Cls Loss | DFL Loss | Tendance |')
print('|-------|---------|----------|----------|----------|----------|')

for ep in range(1, 16):
    if ep in epochs_data:
        data = epochs_data[ep]
        # Calculer la tendance par rapport à l'epoch précédent
        if ep > 1 and ep-1 in epochs_data:
            prev_box = epochs_data[ep-1]['box_loss']
            curr_box = data['box_loss']
            tendance = 'BAISSE' if curr_box < prev_box else 'HAUSSE'
        else:
            tendance = 'START'
        
        print(f"| {ep:5} | {data['gpu']:7.2f} | {data['box_loss']:8.3f} | {data['cls_loss']:8.3f} | {data['dfl_loss']:8.3f} | {tendance:8} |")

print()
if epochs_data:
    first_epoch = min(epochs_data.keys())
    last_epoch = max(epochs_data.keys())
    
    print('=' * 80)
    print('ANALYSE STATISTIQUE')
    print('=' * 80)
    print()
    
    # Calcul des améliorations
    first = epochs_data[first_epoch]
    last = epochs_data[last_epoch]
    
    box_improvement = ((first['box_loss'] - last['box_loss']) / first['box_loss']) * 100
    cls_improvement = ((first['cls_loss'] - last['cls_loss']) / first['cls_loss']) * 100
    dfl_improvement = ((first['dfl_loss'] - last['dfl_loss']) / first['dfl_loss']) * 100
    
    print(f'EVOLUTION EPOCH 1 -> EPOCH {last_epoch}:')
    print(f'   Box Loss:   {first["box_loss"]:.3f} -> {last["box_loss"]:.3f} ({box_improvement:+.1f}%)')
    print(f'   Cls Loss:   {first["cls_loss"]:.3f} -> {last["cls_loss"]:.3f} ({cls_improvement:+.1f}%)')
    print(f'   DFL Loss:   {first["dfl_loss"]:.3f} -> {last["dfl_loss"]:.3f} ({dfl_improvement:+.1f}%)')
    print()
    
    print('MOYENNES:')
    avg_box = sum(d['box_loss'] for d in epochs_data.values()) / len(epochs_data)
    avg_cls = sum(d['cls_loss'] for d in epochs_data.values()) / len(epochs_data)
    avg_dfl = sum(d['dfl_loss'] for d in epochs_data.values()) / len(epochs_data)
    
    print(f'   Box Loss Moyenne:   {avg_box:.3f}')
    print(f'   Cls Loss Moyenne:   {avg_cls:.3f}')
    print(f'   DFL Loss Moyenne:   {avg_dfl:.3f}')
    print()
    
    # Meilleures/pires epochs
    best_epoch = min(epochs_data.keys(), key=lambda e: epochs_data[e]['box_loss'])
    worst_epoch = max(epochs_data.keys(), key=lambda e: epochs_data[e]['box_loss'])
    
    print(f'MEILLEURES PERFORMANCES:')
    print(f'   Meilleur Box Loss: Epoch {best_epoch} ({epochs_data[best_epoch]["box_loss"]:.3f})')
    print(f'   (Pire: Epoch {worst_epoch} ({epochs_data[worst_epoch]["box_loss"]:.3f}))')
