from fastapi import APIRouter, UploadFile, File
from backend.models.inference import predict_image

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await predict_image(file)
    return result