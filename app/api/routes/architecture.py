from fastapi import APIRouter
from app.models.ssl_models import call_model

router = APIRouter()

@router.post("/supervised-learning/{model_name}")
def sl_model(model_name:str):
    return

@router.post("/self-supervised-learning/{model_name}")
def ssl_model(model_name:str='test'):

    model = call_model(model_name)
    print(model.summary())

    return