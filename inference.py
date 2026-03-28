import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import Image
import io

device = torch.device("cpu")

model = EfficientNet.from_name('efficientnet-b7', num_classes=1)

state_dict = torch.load("cardiovision_b7.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    score = torch.sigmoid(output).item()

    return [{"label":"risk","score":score}]