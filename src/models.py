from pydantic import BaseModel, validator

class ImagePredictionResponse(BaseModel):
    class_: int
    confidence: float

class ErrorResponse(BaseModel):
    error: str