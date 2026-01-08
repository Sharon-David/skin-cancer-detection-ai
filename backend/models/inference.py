from PIL import Image
import io
import random

async def predict_image(file):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    classes = ["Melanoma", "Nevus", "Benign Keratosis"]
    prediction = random.choice(classes)

    return {
        "prediction": prediction,
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "note": "Dummy model – replace with trained model"
    }
