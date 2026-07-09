import torch
from PIL import Image
from torchvision import transforms

from model import get_model

# -----------------------
# Class Names
# -----------------------
classes = [
    "Person",
    "Bicycle",
    "Car",
    "Bus",
    "Dog"
]

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Model
# -----------------------
model = get_model()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# Enter Image Path Here
# -----------------------
image_path = "data/val2017/000000035197.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# -----------------------
# Prediction
# -----------------------
with torch.no_grad():

    output = model(image)

    predicted = torch.argmax(output, dim=1).item()

print("Prediction:", classes[predicted])