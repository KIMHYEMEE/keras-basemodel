from app.models.self_supervised.autoencoder import *

def call_model(model_name):
    model = simple_ae(128)

    return model