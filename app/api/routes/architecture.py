from fastapi import APIRouter, HTTPException
from app.models.ssl_models import build_model, ModelNameSSL

import keras
import numpy as np

router = APIRouter()

@router.post("/supervised-learning/{model_name}")
def sl_model(model_name:str):
    return

@router.get("/self-supervised-learning/{model_name}")
def ssl_model(model_name:ModelNameSSL):

    model = build_model(model_name)

    if model != None:
        print(model.model.summary())

        model_info = {'model_name':model_name,
                      'info':model.get_info()}
        
    else:
        raise HTTPException(status_code=404)

    return model_info

@router.get("/self-supervised-learning/{model_name}/test/mnist")
def ssl_mnist(model_name:ModelNameSSL,
              epochs:int=10,
              batch_size:int=None):
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    data_variance = np.var(x_train / 255.0)

    model = build_model(model_name,data_variance)

    if batch_size is None:
        model.model.fit(x_train_scaled,x_train_scaled, epochs=epochs)
    else:
        model.model.fit(x_train_scaled,x_train_scaled, epochs=epochs, batch_size=batch_size)