from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def get_model_list():
    return