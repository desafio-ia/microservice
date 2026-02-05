import os
import torch
from src.model.neural_model import ClassificationModel
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "animal_recognition_model_v2.pth")

class_names = [
    "cow", "goat", "ostrich", "pigeon", "iguana", "possum", "peacock", "cat", "lizard", "horse"   
]

model = ClassificationModel(3, 32, 10)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)
model.eval()

def predict_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    return {
        "modelId": os.getenv("NAME_MODEL", "unknown"),
        "nameSpecies": class_names[predicted.item()],
        "confidence": float(confidence.item())
    }