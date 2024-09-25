from enum import Enum
from app.models.self_supervised.autoencoder import *

class ModelNameSSL(str, Enum):
    sim_ae = 'simple_autoencoder'
    cnn_ae = 'cnn_autoencoder'
    vae = 'variational_autoencoder'
    vq_vae = 'vector_quantized_variational_autoencoder'

def build_model(model_name,data_var=None):

    if model_name == 'simple_autoencoder':
        # ref: https://blog.keras.io/building-autoencoders-in-keras.html
        model = simple_autoencoder((28,28,1))
    
    elif model_name == 'cnn_autoencoder':
        # ref: https://keras.io/examples/vision/autoencoder/
        model = cnn_autoencoder((28,28,1))
    
    elif model_name == 'variational_autoencoder':
        # ref: https://keras.io/examples/generative/vae/
        model = vae((28,28,1))

    elif model_name == 'vector_quantized_variational_autoencoder':
        # ref: https://keras.io/examples/generative/vq_vae/
        model = vq_vae((28,28,1),train_variance=data_var)


    else:
        model = None

    return model