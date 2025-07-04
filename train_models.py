import os
import tensorflow as tf
from keras.applications import ResNet50, VGG16, InceptionV3
from keras.models import Model, save_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint 
# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Reducido para pruebas
LEARNING_RATE = 1e-4

# Rutas
TRAIN_DIR = os.path.join('dataset', 'train')
VAL_DIR = os.path.join('dataset', 'validation')
MODELS_DIR = 'modelos'

def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True)

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    return train_generator, val_generator

def build_model(base_model):
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_gen, val_gen = create_generators()

    models = {
        'resnet50': ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3)),
        'vgg16': VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3)),
        'inceptionv3': InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    }

    for name, base_model in models.items():
        print(f"\n🔧 Entrenando {name}...")
        model = build_model(base_model)
        
        callbacks = [
            EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(MODELS_DIR, f'{name}_custom.h5'),
                monitor='val_accuracy',
                save_best_only=True)
        ]

        history = model.fit(
            train_gen,
            steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=max(1, val_gen.samples // BATCH_SIZE),
            callbacks=callbacks,
            verbose=1)

        # Evaluación final
        val_loss, val_acc, val_prec, val_rec = model.evaluate(val_gen)
        print(f"\n📊 {name.upper()} - Resultados Finales:")
        print(f"Accuracy: {val_acc*100:.2f}%")
        print(f"Precision: {val_prec*100:.2f}%")
        print(f"Recall: {val_rec*100:.2f}%")

if __name__ == "__main__":
    train_models()