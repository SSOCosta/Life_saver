# model_loader.py
from tensorflow.keras.models import load_model

def load_image_classifier_model(model_path):
    return load_model(model_path)