"""
routes.py — API endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from backend.core.utils import validate_image_upload
from backend.models.inference import predict

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read bytes
    file_bytes = await file.read()

    # Validate
    validate_image_upload(file_bytes, file.filename)

    # Run inference
    try:
        result = predict(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return JSONResponse(content=result)
