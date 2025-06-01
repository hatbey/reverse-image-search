# app/model.py
import torch
from torchvision import models, transforms
from PIL import Image

# Load ResNet50 and remove the classifier
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def extract_embedding(img: Image.Image) -> torch.Tensor:
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(img_t)
    return vec.squeeze().numpy()
