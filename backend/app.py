import os
import cv2
import numpy as np
import requests
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
# HUGGINGFACE API CONFIG
# ---------------------------------------------------------

HF_API_URL = "https://router.huggingface.co/hf-inference/models/keshavnayak15/cardiovison-b7"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

@app.on_event("startup")
async def startup():
    await connect_to_mongo()


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
# HUGGINGFACE INFERENCE
# ---------------------------------------------------------

def query_huggingface(image_bytes):

    response = requests.post(
        HF_API_URL,
        headers=headers,
        data=image_bytes
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"HuggingFace inference failed: {response.text}"
        )

    return response.json()


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):

    db = get_database()

    contents = await file.read()

    result = query_huggingface(contents)

    # Example HF response handling
    try:
        risk_score = result[0]["score"]
    except:
        risk_score = 0.5

    prediction_class = "High Risk" if risk_score > 0.6 else "Low Risk"

    confidence = risk_score

    heatmap_filename = None

    pred_doc = {

        "id": str(uuid4()),
        "user_id": current_user["id"],
        "image_filename": file.filename,
        "risk_score": risk_score,
        "confidence": confidence,
        "prediction_class": prediction_class,
        "heatmap_url": "",
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

    report_file = os.path.join(OS_PATH_REPORTS, f"{prediction_id}.pdf")

    pdf.output(report_file)

    return FileResponse(report_file, filename="CardioVision_Report.pdf")