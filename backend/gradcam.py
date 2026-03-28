import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import io
import os
import requests

device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "cardiovision_b7.pth")

# 🔽 Load model
model = EfficientNet.from_name('efficientnet-b7', num_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 🔽 Hook variables
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# 🔽 Attach hooks to LAST conv layer
target_layer = model._conv_head
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# 🔽 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    score = torch.sigmoid(output).item()
    return score


# ---------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------
def generate_gradcam(image_bytes, save_path):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    output = model(tensor)
    score = torch.sigmoid(output)

    model.zero_grad()
    score.backward()

    global gradients, features

    grads = gradients.detach().numpy()[0]
    fmap = features.detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    original = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)

    return save_path