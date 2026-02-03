import os
import torch
from model.neural_model import Neural
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cpu")

class_names = [
    'avião','carro','passáro','gato','veado',
    'cachorro','sapo','cavalo','navio','caminhão'
]

model = Neural(num_classes=10)
model.load_state_dict(
    torch.load("model/neural_cifar10.pt", map_location=DEVICE)
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