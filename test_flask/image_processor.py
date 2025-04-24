import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models
import h5py
from tensorflow.keras.models import load_model
import cv2

"""## Configuration"""

# Chemin vers le dossier contenant les sous-dossiers 'M' et 'NM'
data_dir = '/mnt/sda/Dataset_mini'  # <<< A MODIFIER AVEC TON CHEMIN EXACT

# Paramètres
img_size = (224, 224)
batch_size = 16

# Chargement du dataset
train_ds = image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

class_names = train_ds.class_names
print("Classes détectées:", class_names)

# Normalisation des images avec la préproc VGG16
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))

"""## Creation du modèle"""

# Création du modèle de base (VGG16 sans la classification finale)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Ajout de la classification personnalisée
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(train_ds, epochs=3)

"""## Enregistrement du modèle"""

model.save("/home/malekbennabi/Téléchargements/mon_modele224.h5")
print("Modèle et poids sauvegardés dans 'mon_modele224.h5'")

"""## Inspection du modèle enregistré"""

def load_and_inspect_model(file_path):
    """
    Charge un modèle sauvegardé au format .h5 et affiche des informations sur son contenu,
    son architecture et les poids de chaque couche.

    Args:
        file_path (str): Chemin du fichier .h5 contenant le modèle.
    """
    # Afficher la liste des groupes et datasets dans le fichier .h5
    with h5py.File(file_path, 'r') as file:
        print("Liste des objets dans le fichier .h5:")
        file.visititems(lambda name, obj: print(name, obj))

        print("__________________________________________________________________________________________________________________________________ \n")

        # Afficher les clés du modèle sauvegardé
        print("\nClés du modèle sauvegardé:")
        for key in file.keys():
            print(key)

        print("__________________________________________________________________________________________________________________________________ \n")

    # Charger le modèle depuis le fichier .h5
    model = load_model(file_path)

    # Afficher l'architecture du modèle
    print("Architecture du modèle :")
    model.summary()

    print("__________________________________________________________________________________________________________________________________ \n")

    # Inspecter les poids de chaque couche
    for layer in model.layers:
        weights = layer.get_weights()
        print(f"Poids pour la couche {layer.name}: {weights}")

print(load_and_inspect_model("/home/malekbennabi/Téléchargements/mon_modele224.h5"))

"""## Activation map"""

# Fonction pour afficher les cartes d'activation
def display_activation_maps(model, img_path, layer_names):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Modèle pour extraire les activations
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x)

    for layer_name, activation in zip(layer_names, activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]

        n_cols = 8
        n_rows = n_features // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
        fig.suptitle(f"Activations - Layer: {layer_name}", fontsize=16)

        for i in range(n_rows * n_cols):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(activation[0, :, :, i], cmap='viridis')
            ax.axis('off')

        plt.tight_layout()

        plt.savefig(f"/home/malekbennabi/Téléchargements/Activation maps/activation_maps_{layer_name}.png")
        plt.savefig(f"/home/malekbennabi/Téléchargements/Activation maps/activation_maps_{layer_name}.svg")

        #plt.show()

"""## GRADCAM superposition d'image"""

def make_gradcam_heatmap(img_path, model, conv_layer_name='block5_conv3'):
    # Préparation de l’image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Accès à la couche imbriquée dans 'vgg16'
    base_model = model.get_layer('vgg16')
    conv_layer = base_model.get_layer(conv_layer_name)

    # Création du modèle Grad-CAM
    grad_model = Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
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

# Affichage de la heatmap superposée

def display_gradcam(img_path, model, last_conv_layer_name='block5_conv3'):
    heatmap, img = make_gradcam_heatmap(img_path, model, last_conv_layer_name)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * 0.4 + img
    superimposed_img = np.uint8(superimposed_img)

    # Affichage avec colorbar
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(superimposed_img.astype("uint8"))
    ax.axis('off')
    ax.set_title("Grad-CAM avec échelle de couleurs", fontsize=14)

    # Ajout d’une colorbar à côté
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance relative', rotation=270, labelpad=15)

    plt.savefig('static/img/gradcam.png')
    #plt.show()


"""## Afficher toutes les couches du modèle enregistré"""

import tensorflow as tf
from tensorflow.keras.models import load_model


if __name__ == '__main__':

    display_gradcam('/home/malekbennabi/Images/Captures d’écran/Capture31.png', base_model, last_conv_layer_name='block5_conv3')


