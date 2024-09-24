from app.models.self_supervised.autoencoder import *

def build_model(model_name):

    if model_name == 'simple_autoencoder':
        model = simple_autoencoder(128)

    else:
        model = None

    return model