import os
import torch
from src.model.neural_model import Neural
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "neural_cifar10.pt")

class_names = [
    'avião','carro','passáro','gato','veado',
    'cachorro','sapo','cavalo','navio','caminhão'
]

model = Neural(num_classes=10)
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