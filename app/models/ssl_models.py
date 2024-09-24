from enum import Enum
from app.models.self_supervised.autoencoder import *

class ModelNameSSL(str, Enum):
    sim_ae = 'simple_autoencoder'
    cnn_ae = 'cnn_autoencoder'

def build_model(model_name):

    if model_name == 'simple_autoencoder':
        # ref: https://blog.keras.io/building-autoencoders-in-keras.html
        model = simple_autoencoder(128)

    else:
        model = None

    return model