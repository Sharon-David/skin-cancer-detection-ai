"""
routes.py — API endpoints with Grad-CAM, OOD detection, uncertainty estimation
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from backend.core.utils import validate_image_upload
from backend.models.inference import predict, predict_with_uncertainty
from backend.models.explainability import generate_gradcam
from backend.models.ood_detection import is_ood
from backend.models.inference import get_model

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    gradcam: bool = True,
    uncertainty: bool = True,
):
    # Read bytes
    file_bytes = await file.read()

    # Validate
    validate_image_upload(file_bytes, file.filename)

    # OOD check first
    ood_result = is_ood(file_bytes)
    if ood_result["is_ood"]:
        return JSONResponse(content={
            "is_ood":         True,
            "ood_reasons":    ood_result["reasons"],
            "prediction":     None,
            "confidence":     None,
            "prob_malignant": None,
            "prob_benign":    None,
            "risk_level":     "Unknown",
            "recommendation": "This image does not appear to be a skin lesion image. Please upload a clear dermoscopy or clinical photo of a skin lesion.",
            "disclaimer":     "This is a research tool only. Not a medical device.",
            "gradcam_image":  None,
        })

    # Run inference with uncertainty
    try:
        if uncertainty:
            result = predict_with_uncertainty(file_bytes)
        else:
            result = predict(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Generate Grad-CAM heatmap
    if gradcam:
        try:
            model = get_model()
            heatmap = generate_gradcam(model, file_bytes)
            result["gradcam_image"] = heatmap
        except Exception:
            result["gradcam_image"] = None

    result["is_ood"]      = False
    result["ood_reasons"] = []

    return JSONResponse(content=result)
