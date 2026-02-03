from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

def load_and_preprocess_image(file):
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        raise ValueError("Invalid image")

    return transform(image).unsqueeze(0)
