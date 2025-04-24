import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory

def train_and_save_model(data_dir, output_path='mon_modele224.h5'):
    img_size = (224, 224)
    batch_size = 16

    # Chargement du dataset
    train_ds = image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))

    # Modèle de base
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    # API Functional
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, epochs=3)

    model.save(output_path)
    print(f"Modèle sauvegardé dans {output_path}")
    return model

def load_and_inspect_model(file_path):
    """
    Charge et affiche des infos sur un modèle Keras sauvegardé.
    """
    with h5py.File(file_path, 'r') as file:
        print("Contenu du fichier .h5 :")
        file.visititems(lambda name, obj: print(name, obj))
        print("Clés du fichier :", list(file.keys()))

    model = tf.keras.models.load_model(file_path)
    print("Résumé du modèle :")
    model.summary()

    return model
