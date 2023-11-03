from datetime import datetime

from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from .models import ImagePredictionResponse, ErrorResponse
import torch
import torchvision.transforms as transforms
from torch.nn.functional import softmax

import io

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate the FastAPI application
app = FastAPI()

img_model = torch.jit.load("src/CNNModel.pt")

img_model.eval()
logger.info("Model loaded successfully")


class RemoveAlphaChannel:
    def __call__(self, image):
        return image.convert("RGB")

@app.post('/predict')
async def classify_image(file: UploadFile = None):
    logger.info("Entered classify_image endpoint")
    if not file:
        logger.warning("No file object received!")
        return JSONResponse(content=ErrorResponse(error="no file").model_dump(), status_code=400)
    elif file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        logger.warning("File received, but not a valid image type (jpeg, jpg, png))")
        return JSONResponse(content=ErrorResponse(error="Invalid file type").model_dump(), status_code=400) 
    else:
        logger.info(f"File object exists. Filename: {file.filename}, Content type: {file.content_type}")
        
    
    # Read image contents
    image_data = await file.read()

    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    logger.info(f"Image read successfully")

    # Transform image, hyperparameters coded in from training
    means = [0.0013, 0.0014, 0.0011]
    stds = [0.0030, 0.0030, 0.0030]

    preprocess  = transforms.Compose([
        RemoveAlphaChannel(),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(means, stds,inplace=True),
        ])
        
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    logger.info(f"Image transformed successfully")

    with torch.no_grad():
        output = img_model(input_batch)
    
    logger.info(f"Model prediction completed successfully")
    
    probabilities = softmax(output, dim=-1)
    prob, predicted_class = torch.max(probabilities, 1)
    logger.info(f"Prediction made successfully. Predicted class: {predicted_class.item()}, confidence: {prob.item()}")

    return ImagePredictionResponse(class_=predicted_class.item(), confidence=prob.item())


@app.get("/health")
def get_current_time():
    return {"time": datetime.utcnow().isoformat()}
