from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2

from werkzeug.utils import secure_filename
from test import display_gradcam_filtered, make_gradcam_heatmap
from tensorflow.keras.applications.vgg16 import VGG16

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/outputs'

# Charger le modèle VGG16 une seule fois
model = VGG16(weights='imagenet')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Générer la heatmap Grad-CAM
            heatmap, original = make_gradcam_heatmap(img_path, model, last_conv_layer_name='block5_conv3')

            heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

            # Sauvegarde des images
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], f'gradcam_{filename}')
            cv2.imwrite(original_path, original[..., ::-1])
            cv2.imwrite(gradcam_path, superimposed)

            display_gradcam_filtered(img_path, model, last_conv_layer_name='block5_conv3', threshold=0.6)
            threshold_percentage = int(0.6 * 100)

            return render_template('index.html',
                                   original_image=url_for('static', filename=f'outputs/original_{filename}'),
                                   output_image=url_for('static', filename=f'outputs/gradcam_{filename}'),
                                   filtered_image=url_for('static', filename='img/gradcam_filtre.png'),
                                   threshold_percentage=threshold_percentage)

    return render_template('index.html', original_image=None, output_image=None, filtered_image=None)

# Route pour classification.html
@app.route('/classification')
def classification():
    return render_template('classification.html')

# Route pour segmentation.html
@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
