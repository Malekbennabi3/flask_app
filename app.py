from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename

from test import display_gradcam_filtered
from test import make_gradcam_heatmap_tflite  # on va l'adapter aussi

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/outputs'

# Charger modèle TFLite une fois
interpreter = tf.lite.Interpreter(model_path="vgg16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Générer heatmap Grad-CAM (adapté pour TFLite)
            heatmap, original = make_gradcam_heatmap_tflite(img_path, interpreter, input_details, output_details, 'block5_conv3')

            # Superposer heatmap à l'image
            heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

            # Sauvegarder les images
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], f'gradcam_{filename}')
            cv2.imwrite(original_path, original[..., ::-1])  # RGB → BGR
            cv2.imwrite(gradcam_path, superimposed)

            # Affichage filtré (si nécessaire)
            display_gradcam_filtered(img_path, interpreter, input_details, output_details, last_conv_layer_name='block5_conv3', threshold=0.6)

            threshold_percentage = int(0.6 * 100)

            return render_template('index.html',
                                   original_image=url_for('static', filename=f'outputs/original_{filename}'),
                                   output_image=url_for('static', filename=f'outputs/gradcam_{filename}'),
                                   filtered_image=url_for('static', filename='img/gradcam_filtre.png'),
                                   threshold_percentage=threshold_percentage)

    return render_template('index.html', original_image=None, output_image=None, filtered_image=None)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
