import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

def make_gradcam_heatmap(img_path, model, last_conv_layer_name='block5_conv3'):
    # Préparation de l’image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Aller chercher la vraie couche dans vgg16
    base_model = model.get_layer('vgg16')
    conv_layer_output = base_model.get_layer(last_conv_layer_name).output

    # Re-construire un modèle avec cette sortie intermédiaire
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_layer_output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), img

def display_gradcam(img_path, model, last_conv_layer_name='block5_conv3', save_path='static/img/gradcam.png'):
    heatmap = make_gradcam_heatmap(img_path, model, last_conv_layer_name)
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * 0.4 + img
    superimposed_img = np.uint8(superimposed_img)

    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title('Grad-CAM')

    plt.savefig(save_path)
