import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import h5py
from tensorflow.keras.preprocessing import image

img_size = (224, 224)  # Expected input size for VGG16


def display_all_layers(model, prefix=''):
    for i, layer in enumerate(model.layers):
        print(f"{prefix}[{i}] {layer.name} --- {layer.__class__.__name__}")
        if isinstance(layer, tf.keras.Model):
            display_all_layers(layer, prefix + '  ')


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

def display_layer_names(model_path):
    """
    Affiche les noms de toutes les couches d'un modèle Keras pré-entraîné
    sauvegardé au format .h5.

    Args:
        model_path (str): Le chemin vers le fichier .h5 du modèle.
    """
    try:
        model = load_model(model_path)
        print(f"Noms des couches du modèle chargé depuis : {model_path}")
        for i, layer in enumerate(model.layers):
            print(f"[{i}] : {layer.name}--- {layer.output.name}---{layer}")
    except FileNotFoundError:
        print(f"Erreur: Le fichier de modèle au chemin '{model_path}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input


def make_gradcam_heatmap(img_path, model, last_conv_layer_name='block5_conv3', img_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    try:
        base_model = model.get_layer('vgg16')
        conv_layer = base_model.get_layer(last_conv_layer_name).output
        grad_model = Model(inputs=base_model.input, outputs=[conv_layer, base_model.output])
    except ValueError:
        conv_layer = model.get_layer(last_conv_layer_name).output
        grad_model = Model(inputs=model.input, outputs=[conv_layer, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if len(predictions.shape) == 2:
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]
        else:
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

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap onto image
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

    # Affichage avec colorbar réelle (image séparée pour la légende)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(superimposed_img[..., ::-1])
    ax.axis('off')
    ax.set_title(f"Grad-CAM: {last_conv_layer_name}")

    # Créer une image fictive pour la colorbar avec les bonnes valeurs de heatmap
    cbar_img = ax.imshow(heatmap_resized, cmap='jet', alpha=1.0, visible=False)  # cachée mais valeurs réelles

    # Ajouter colorbar correctement
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    fig.colorbar(cbar_img, cax=cax)
    plt.tight_layout()

    plt.savefig("static/img/gradcam_result.png")
    plt.show()





def save_and_display_heatmap(img_path, heatmap, alpha=0.4):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    # Save and display
    output_path = "static/img/gradcam_result.jpg"
    cv2.imwrite(output_path, superimposed_img)
    print(f"Saved Grad-CAM result to: {output_path}")

    # Show image (optional)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM")
    #plt.show()


# Fonction pour afficher les cartes d'activation
def display_activation_maps(model, img_path, layer_names, img_size=(224, 224)):
    # Chargement et prétraitement de l'image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Ajout dimension batch
    x = preprocess_input(x)

    # Accès au sous-modèle VGG16
    base_model = model.get_layer('vgg16')

    # Création du modèle pour extraire les activations
    layer_outputs = [base_model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=base_model.input, outputs=layer_outputs)

    # Prédiction
    activations = activation_model.predict(x)

    # Si une seule couche est demandée, activations est un array (pas une liste)
    if not isinstance(activations, list):
        activations = [activations]

    # Affichage des activations
    for layer_name, activation in zip(layer_names, activations):
        # S'assurer qu'on a bien 4 dimensions (batch, height, width, channels)
        if len(activation.shape) != 4:
            print(f"⚠️ La couche '{layer_name}' ne retourne pas de cartes 2D (activation.shape = {activation.shape})")
            continue

        n_features = activation.shape[-1]
        size = activation.shape[1]

        n_cols = 8
        n_rows = n_features // n_cols if n_features % n_cols == 0 else (n_features // n_cols) + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        fig.suptitle(f"Activations - Layer: {layer_name}", fontsize=16)

        # Gestion du cas où axes est 1D (n_rows == 1)
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(n_rows * n_cols):
            ax = axes[i // n_cols, i % n_cols]
            if i < n_features:
                ax.imshow(activation[0, :, :, i], cmap='viridis')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"activation_maps_{layer_name}.png")
        plt.show()



def display_gradcam_filtered(img_path, model, last_conv_layer_name='block5_conv3', threshold=0.6, img_size=(224, 224)):
    # Générer heatmap avec ta fonction existante
    heatmap, _ = make_gradcam_heatmap(img_path, model, last_conv_layer_name, img_size)

    # Charger image d’origine (pour dimensions)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Redimensionner heatmap à la taille de l’image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Appliquer un masque basé sur le seuil
    mask = heatmap_resized >= threshold
    masked_heatmap = np.zeros_like(heatmap_resized)
    masked_heatmap[mask] = heatmap_resized[mask]

    # Affichage et sauvegarde avec colorbar visible
    plt.figure(figsize=(8, 8))
    img_display = plt.imshow(masked_heatmap, cmap='jet')
    plt.colorbar(img_display, label='Importance relative')
    plt.title(f"Zones fortement activées (seuil: {int(threshold * 100)}%)")
    plt.axis('off')

    os.makedirs("static/img", exist_ok=True)
    plt.savefig('static/img/gradcam_filtre.png')
    plt.close()



if __name__ == '__main__':

    model = load_model("Backend/models/modele_vgg16_224.h5")

    print(load_and_inspect_model("Backend/models/modele_vgg16_224.h5"))



    # Chemin vers ton image
    img_path = 'static/img/Capture.png'

    # Charger et préparer l'image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Effectuer l'inférence
    predictions = model.predict(x)


    # Afficher les résultats
    print(f'Prédictions pour l\'image : {predictions[0]} ')



    print(f'Couches du modele :\n')
    display_layer_names("Backend/models/modele_vgg16_224.h5")




    print("Structure complète du modèle :")
    display_all_layers(model)



    display_activation_maps(model, img_path, ['block3_conv3'])

    show_gradcam(img_path, model, last_conv_layer_name='block5_conv3')

    display_gradcam_filtered(img_path, model, last_conv_layer_name='block5_conv3', threshold=0.6)
