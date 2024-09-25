from fastapi import APIRouter, HTTPException
from app.models.ssl_models import build_model, ModelNameSSL

import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
              epochs:int=1,
              batch_size:int=128):
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0)
    x_test_scaled = (x_test / 255.0)

    data_variance = np.var(x_train / 255.0)

    model = build_model(model_name,data_variance)

    if batch_size is None:
        model.model.fit(x_train_scaled,x_train_scaled, epochs=epochs)
    else:
        model.model.fit(x_train_scaled,x_train_scaled, epochs=epochs, batch_size=batch_size, shuffle=True)

    reconstructed_img = model.model.predict(x_train_scaled[:1])[0]
    if len(reconstructed_img.shape) > 3:
        reconstructed_img = reconstructed_img[0]
    test_img = x_train_scaled[0]

    show_subplot(test_img,reconstructed_img,model_name)


def show_subplot(original, reconstructed,model_name):
    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(reconstructed)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.savefig(f'./app/fig/result-{model_name}.png')
    plt.close()
