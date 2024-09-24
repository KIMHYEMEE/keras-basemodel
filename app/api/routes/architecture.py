from fastapi import APIRouter

router = APIRouter()

@router.post("/supervised-learning/{model_name}")
def sl_model(model_name:str):
    return

@router.post("/self-supervised-learning/{model_name}")
def ssl_model(model_name:str):
    return