from fastapi import APIRouter
from app.models.ssl_models import ModelNameSSL

router = APIRouter()

@router.post("/")
def get_model_list():
    ssl_names = [k.value for k in ModelNameSSL]

    return {'self-supervised':ssl_names}