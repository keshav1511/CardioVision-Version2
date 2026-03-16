import os
import cv2
import torch
import numpy as np
import requests
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from jose import JWTError, jwt

from backend.auth import verify_password, get_password_hash, create_access_token, ALGORITHM, SECRET_KEY
from backend.models import UserCreate, UserResponse, Token, PredictionResult
from backend.database import connect_to_mongo, close_mongo_connection, get_database


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(title="CardioVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# ---------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

OS_PATH_HEATMAPS = os.path.join(BASE_DIR, "heatmaps")
OS_PATH_REPORTS = os.path.join(BASE_DIR, "reports")

os.makedirs(OS_PATH_HEATMAPS, exist_ok=True)
os.makedirs(OS_PATH_REPORTS, exist_ok=True)


# ---------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------

MODEL_PATH = "cardiovision_b7.pth"

MODEL_URL = "https://huggingface.co/spaces/keshavnayak15/cardiovision-b7/resolve/main/cardiovision_b7.pth"

device = torch.device("cpu")

model = None
target_layer = None


# ---------------------------------------------------------
# DOWNLOAD MODEL
# ---------------------------------------------------------

def download_model():

    if os.path.exists(MODEL_PATH):
        print("Model already exists")
        return

    print("Downloading EfficientNet-B7 model...")

    r = requests.get(MODEL_URL, stream=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Model downloaded successfully")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def load_model():

    global model
    global target_layer

    download_model()

    print("Loading model...")

    model = models.efficientnet_b7(weights=None)

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, 1
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    target_layer = model.features[-1]

    print("Model loaded successfully")


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

@app.on_event("startup")
async def startup():

    await connect_to_mongo()

    load_model()


@app.on_event("shutdown")
async def shutdown():

    await close_mongo_connection()


# ---------------------------------------------------------
# AUTH
# ---------------------------------------------------------

async def get_current_user(token: str = Depends(oauth2_scheme)):

    db = get_database()

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    try:

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        email = payload.get("sub")

        if email is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = await db.users.find_one({"email": email})

    if user is None:
        raise credentials_exception

    return user


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

@app.get("/")
def root():

    return {"message": "CardioVision API Running"}


# ---------------------------------------------------------
# SIGNUP
# ---------------------------------------------------------

@app.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate):

    db = get_database()

    existing = await db.users.find_one({"email": user.email})

    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = get_password_hash(user.password)

    user_id = str(uuid4())

    user_doc = {
        "id": user_id,
        "name": user.name,
        "email": user.email,
        "hashed_password": hashed,
        "created_at": datetime.utcnow()
    }

    await db.users.insert_one(user_doc)

    return UserResponse(
        id=user_id,
        name=user.name,
        email=user.email
    )


# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):

    db = get_database()

    user = await db.users.find_one({"email": form_data.username})

    if not user or not verify_password(form_data.password, user["hashed_password"]):

        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
        )

    token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=timedelta(days=7)
    )

    return {"access_token": token, "token_type": "bearer"}


# ---------------------------------------------------------
# GRADCAM
# ---------------------------------------------------------

def generate_gradcam(input_tensor):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)

    score = torch.sigmoid(output)

    model.zero_grad()

    score.backward(torch.ones_like(score))

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * acts, dim=1)

    cam = F.relu(cam)

    cam = cam.squeeze()

    cam = cam.detach().cpu().numpy()

    cam = cam - cam.min()

    if cam.max() != 0:
        cam = cam / cam.max()

    handle_forward.remove()
    handle_backward.remove()

    return cam


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):

    db = get_database()

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(image_rgb)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    tensor = transform(image_pil).unsqueeze(0).to(device)

    output = model(tensor)

    risk_score = torch.sigmoid(output).item()

    prediction_class = "High Risk" if risk_score > 0.6 else "Low Risk"

    confidence = risk_score


    cam = generate_gradcam(tensor)

    cam = cv2.resize(
        cam,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    cam = cv2.GaussianBlur(cam, (31, 31), 0)

    cam = cam - cam.min()

    cam = cam / (cam.max() + 1e-8)

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)


    heatmap_filename = f"heatmap_{uuid4()}.jpg"

    heatmap_path = os.path.join(OS_PATH_HEATMAPS, heatmap_filename)

    cv2.imwrite(heatmap_path, overlay)


    pred_doc = {

        "id": str(uuid4()),
        "user_id": current_user["id"],
        "image_filename": file.filename,
        "risk_score": risk_score,
        "confidence": confidence,
        "prediction_class": prediction_class,
        "heatmap_url": f"/heatmaps/{heatmap_filename}",
        "created_at": datetime.utcnow()

    }

    await db.predictions.insert_one(pred_doc)

    return pred_doc


# ---------------------------------------------------------
# HEATMAP
# ---------------------------------------------------------

@app.get("/heatmaps/{filename}")
async def heatmap(filename: str):

    path = os.path.join(OS_PATH_HEATMAPS, filename)

    return FileResponse(path)


# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

@app.get("/download-report")
async def report(prediction_id: str, current_user: dict = Depends(get_current_user)):

    db = get_database()

    from fpdf import FPDF

    pred = await db.predictions.find_one({
        "id": prediction_id,
        "user_id": current_user["id"]
    })

    if not pred:
        raise HTTPException(status_code=404)

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=16)

    pdf.cell(200,10,"CardioVision Medical Report", ln=True)

    pdf.set_font("Arial", size=12)

    pdf.cell(200,10,f"Patient: {current_user['name']}", ln=True)

    pdf.cell(200,10,f"Risk Score: {pred['risk_score']*100:.2f}%", ln=True)

    pdf.cell(200,10,f"Prediction: {pred['prediction_class']}", ln=True)

    heatmap_path = os.path.join(
        OS_PATH_HEATMAPS,
        os.path.basename(pred["heatmap_url"])
    )

    if os.path.exists(heatmap_path):

        pdf.image(heatmap_path, x=10, y=60, w=120)

    report_file = os.path.join(OS_PATH_REPORTS, f"{prediction_id}.pdf")

    pdf.output(report_file)

    return FileResponse(report_file, filename="CardioVision_Report.pdf")