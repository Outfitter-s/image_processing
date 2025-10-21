from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove
import io
import os
import tensorflow as tf
from shared import BASE_DIR, CSV_PATH, IMAGES_DIR, OUTPUT_H5_PATH

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}}) # TODO: restrict in production

@app.route('/remove', methods=['POST'])
def remove_background():
    """
    POST /remove
    form-data: file -> image to process
    returns: PNG image with background removed
    """
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    uploaded = request.files['file']
    data = uploaded.read()
    try:
        out = remove(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return send_file(io.BytesIO(out), mimetype='image/png')

_model = None

def load_classification_model():
    global _model
    if _model is None:
        if not os.path.exists(OUTPUT_H5_PATH):
            raise FileNotFoundError(f"Model file not found at {OUTPUT_H5_PATH}. Create it using ai/imageClassification/main.py")
        _model = tf.keras.models.load_model(OUTPUT_H5_PATH)

@app.route('/classify', methods=['POST'])
def classify_image():
    """
    POST /classify
    form-data: file -> image to classify
    returns: JSON labelIndex: int
    """
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    uploaded = request.files['file']
    data = uploaded.read()
    try:
        load_classification_model()
    except Exception as e:
        return jsonify({'error': f"model load error: {e}"}), 500

    try:
        # decode and preprocess
        img = tf.io.decode_image(data, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)  # batch dim

        preds = _model.predict(img)
        label_index = int(tf.argmax(preds[0]).numpy())

        # cleanup
        tf.keras.backend.clear_session()

        return jsonify(label_index)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
