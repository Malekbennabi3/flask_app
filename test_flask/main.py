from model_utils import load_and_inspect_model
from gradcam_utils import display_gradcam
import numpy as np

MODEL_PATH = '/home/malekbennabi/Téléchargements/mon_modele224.h5'

# On charge le modèle une seule fois (pas à chaque requête)
model = load_and_inspect_model(MODEL_PATH)

dummy_input = np.zeros((1, 224, 224, 3))  # ou la taille que ton modèle attend
_ = model.predict(dummy_input)  # permet de définir model.input / model.output


def process_image(img_path):
    # Applique le Grad-CAM et sauvegarde dans static/img
    display_gradcam(img_path, model)
