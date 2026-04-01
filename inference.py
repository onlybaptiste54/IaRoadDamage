import os
import cv2
from ultralytics import YOLO

def main():
    # Chemin du meilleur modèle (100 epochs) ✅
    weights_path = "/usr/src/app/runs/detect/Aetheria_RDD/v11_m_rdd_v2_100ep/weights/best.pt"
    
    print(f"🔍 Vérification modèle: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"❌ Modèle non trouvé!")
        return
    
    print(f"✅ Modèle trouvé, chargement...")
    
    # Charger modèle
    try:
        model = YOLO(weights_path)
        print(f"✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return
    
    # Lister les images dispo
    test_dir = "/usr/src/app/RDD_SPLIT/test/images"
    print(f"\n🔍 Recherche images dans: {test_dir}")
    
    if not os.path.exists(test_dir):
        print(f"❌ Dossier images inexistant!")
        return
    
    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Trouvé {len(images)} images")
    
    if not images:
        print("❌ Pas d'images!")
        return
    
    # Tester sur PLUSIEURS images
    for img_name in images[:3]:  # Test sur les 3 premières
        test_image = os.path.join(test_dir, img_name)
        print(f"\n📸 Inférence sur: {img_name}")
        
        try:
            # Inférence avec confidence très basse pour voir tout
            results = model.predict(source=test_image, conf=0.1, imgsz=512)
            
            # Résultats
            for result in results:
                print(f"   Nombre de détections: {len(result.boxes)}")
                
                if len(result.boxes) == 0:
                    print(f"   ⚠️  Aucune détection!")
                else:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        classes = {
                            0: "longitudinal crack",
                            1: "transverse crack",
                            2: "alligator crack",
                            3: "other corruption",
                            4: "pothole"
                        }
                        class_name = classes.get(class_id, f'Class_{class_id}')
                        print(f"   ✅ {class_name}: {confidence:.3f}")
                
                # Sauvegarder
                result.save(f"/usr/src/app/detection_{img_name}")
                print(f"   💾 Sauvegardé: detection_{img_name}")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

if __name__ == "__main__":
    main()

