from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
def load_and_preprocess_image(file):
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        raise ValueError("Invalid image")

    return transform(image).unsqueeze(0)
