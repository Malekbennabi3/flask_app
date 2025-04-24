import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from tensorflow.keras.preprocessing import image


# Charger le modèle
def load_and_inspect_model(file_path):
    try:
        model = load_model(file_path)
        print("Model Architecture:")
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Visualisation de Grad-CAM
def make_gradcam_heatmap(img_path, model, last_conv_layer_name='block5_conv3', img_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Récupérer la couche de convolution
    conv_layer = model.get_layer(last_conv_layer_name).output
    grad_model = Model(inputs=model.input, outputs=[conv_layer, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    img_rgb = tf.keras.preprocessing.image.array_to_img(img_array[0])
    return heatmap.numpy(), np.array(img_rgb)


def show_gradcam(img_path, model, last_conv_layer_name='block5_conv3', img_size=(224, 224), alpha=0.4):
    heatmap, original_img = make_gradcam_heatmap(img_path, model, last_conv_layer_name, img_size)

    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(superimposed_img[..., ::-1])
    ax.axis('off')
    ax.set_title(f"Grad-CAM: {last_conv_layer_name}")

    cbar_img = ax.imshow(heatmap_resized, cmap='jet', alpha=1.0, visible=False)
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    fig.colorbar(cbar_img, cax=cax)
    plt.tight_layout()

    plt.savefig("static/img/gradcam_result.png")
    plt.show()


# Exemple d'utilisation du modèle
if __name__ == '__main__':
    model = load_and_inspect_model("Backend/models/modele_vgg16_224.h5")

    # Préparation de l'image
    img_path = 'static/img/Capture.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    print(f'Predictions for image: {predictions[0]}')

    # Affichage des résultats Grad-CAM
    show_gradcam(img_path, model)
