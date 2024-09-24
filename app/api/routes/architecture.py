from fastapi import APIRouter
from app.models.ssl_models import build_model

router = APIRouter()

@router.post("/supervised-learning/{model_name}")
def sl_model(model_name:str):
    return

@router.post("/self-supervised-learning/{model_name}")
def ssl_model(model_name:str='simple_autoencoder'):

    model = build_model(model_name)

    if model != None:
        print(model.model.summary())

        if 'encoder' in model_name:
            print(model.encoder().summary())

    return