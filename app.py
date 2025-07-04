from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model  # Cambiado a tensorflow.keras
from tensorflow.keras.preprocessing import image
from PIL import Image
from waitress import serve  # Importación para producción

app = Flask(__name__)

# Configuración
MODELS_DIR = 'modelos'
IMG_SIZE = (224, 224)

# Cargar modelos al iniciar
try:
    MODELS = {
        'resnet50': load_model(os.path.join(MODELS_DIR, 'resnet50_custom.h5')),
        'vgg16': load_model(os.path.join(MODELS_DIR, 'vgg16_custom.h5')),
        'inceptionv3': load_model(os.path.join(MODELS_DIR, 'inceptionv3_custom.h5'))
    }
    print("✅ Modelos cargados correctamente")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    MODELS = None

def preprocess_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODELS is None:
        return jsonify({'error': 'Modelos no cargados'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        # Guardar temporalmente
        temp_dir = os.path.join('static', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        # Preprocesar
        img_array = preprocess_image(temp_path)

        # Predecir
        predictions = {}
        for name, model in MODELS.items():
            pred = model.predict(img_array)[0][0]
            predictions[name] = {
                'class': 'Perro' if pred > 0.5 else 'Gato',
                'confidence': float(pred if pred > 0.5 else 1 - pred),
                'raw_score': float(pred)
            }

        return jsonify({
            'success': True,
            'predictions': predictions,
            'image_url': f'/static/temp/{file.filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configuración para producción (Render)
    port = int(os.environ.get("PORT", 10000))  # Usa el puerto de Render o 10000 por defecto
    serve(app, host="0.0.0.0", port=port)  # Servidor Waitress para producción