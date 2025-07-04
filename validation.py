import os
import shutil
from sklearn.model_selection import train_test_split

def actualizar_validation_set(dataset_dir, val_size=0.2):
    """
    Reorganiza automáticamente el dataset manteniendo la proporción de validación
    :param dataset_dir: Ruta al dataset principal (debe contener train/ y validation/)
    :param val_size: Proporción para validation (0.2 = 20%)
    """
    # Preparar directorios temporales
    temp_dir = os.path.join(dataset_dir, 'temp_reorganizacion')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Mover todo a temporal
    for clase in ['dogs', 'cats']:
        # Crear carpeta temporal para la clase
        temp_class_dir = os.path.join(temp_dir, clase)
        os.makedirs(temp_class_dir, exist_ok=True)
        
        # Mover imágenes de train
        train_src = os.path.join(dataset_dir, 'train', clase)
        for img in os.listdir(train_src):
            shutil.move(os.path.join(train_src, img), os.path.join(temp_class_dir, img))
        
        # Mover imágenes de validation
        val_src = os.path.join(dataset_dir, 'validation', clase)
        for img in os.listdir(val_src):
            shutil.move(os.path.join(val_src, img), os.path.join(temp_class_dir, img))
    
    # Recrear estructura vacía
    shutil.rmtree(os.path.join(dataset_dir, 'train'))
    shutil.rmtree(os.path.join(dataset_dir, 'validation'))
    os.makedirs(os.path.join(dataset_dir, 'train', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'train', 'cats'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'validation', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'validation', 'cats'), exist_ok=True)
    
    # Redistribuir las imágenes
    for clase in ['dogs', 'cats']:
        temp_class_dir = os.path.join(temp_dir, clase)
        imagenes = os.listdir(temp_class_dir)
        
        # Dividir según la proporción deseada
        train_imgs, val_imgs = train_test_split(imagenes, test_size=val_size, random_state=42)
        
        # Mover a train
        for img in train_imgs:
            shutil.move(
                os.path.join(temp_class_dir, img),
                os.path.join(dataset_dir, 'train', clase, img)
            )
        
        # Mover a validation
        for img in val_imgs:
            shutil.move(
                os.path.join(temp_class_dir, img),
                os.path.join(dataset_dir, 'validation', clase, img)
            )
    
    # Limpiar temporal
    shutil.rmtree(temp_dir)
    print("✅ Validation set actualizado correctamente!")
    print(f"Proporción actual: {1-val_size:.0%} train | {val_size:.0%} validation")

# Uso:
actualizar_validation_set(
    dataset_dir="C:/python/Clasificador_Perros_Gatos/dataset",
    val_size=0.2  # 20% para validation
)