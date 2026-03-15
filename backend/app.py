import os
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import EmailStr
import numpy as np

# Stub libraries for image proc / report
import cv2 

from auth import verify_password, get_password_hash, create_access_token, ALGORITHM, SECRET_KEY
from models import UserCreate, UserLogin, UserResponse, Token, PredictionResult
from database import connect_to_mongo, close_mongo_connection, db

from jose import JWTError, jwt

app = FastAPI(title="CardioVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, adjust frontend host here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Directory setup
OS_PATH_HEATMAPS = os.path.join(os.path.dirname(__file__), "heatmaps")
OS_PATH_REPORTS = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(OS_PATH_HEATMAPS, exist_ok=True)
os.makedirs(OS_PATH_REPORTS, exist_ok=True)

MODEL_HINT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cardiovision_b7.pth")

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()
    if not os.path.exists(MODEL_HINT_PATH):
        print(f"WARNING: Model file not found at {MODEL_HINT_PATH}. Prediction will run in stub mode.")

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    user_doc = await db.users.find_one({"email": email})
    if user_doc is None:
        raise credentials_exception
    return user_doc

@app.get("/")
def read_root():
    return {"message": "Welcome to CardioVision API"}

@app.get("/health")
def health_check():
    return {"status": "ok", "db_connected": db is not None, "model_exists": os.path.exists(MODEL_HINT_PATH)}

@app.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    user_id = str(uuid4())
    user_dict = {
        "id": user_id,
        "name": user.name,
        "email": user.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    await db.users.insert_one(user_dict)
    return user_dict

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    user_doc = await db.users.find_one({"email": form_data.username})
    if not user_doc or not verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=60*24*7)
    access_token = create_access_token(
        data={"sub": user_doc["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=PredictionResult)
async def predict_risk(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    # Validate file size (10MB) & Type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    file.file.seek(0)
    
    # Read image contents
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    
    # ---------------------------------------------------------
    # In a real scenario, we load cardiovision_b7.pth here.
    # Because this is a generation task without the user's actual weights,
    # we simulate the Grad-CAM generation and Risk Prediction output.
    # ---------------------------------------------------------
    
    import random
    risk_score = random.uniform(0.1, 0.99)
    prediction_class = "High Risk" if risk_score > 0.6 else "Low Risk"
    confidence = random.uniform(0.8, 0.99)
    
    # Simulate saving a heatmap
    heatmap_filename = f"heatmap_{uuid4()}.jpg"
    heatmap_path = os.path.join(OS_PATH_HEATMAPS, heatmap_filename)
    # create a dummy colored image
    dummy_heatmap = cv2.applyColorMap(np.uint8(255 * np.random.rand(256, 256)), cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, dummy_heatmap)
    
    pred_result = {
        "id": str(uuid4()),
        "user_id": current_user["id"],
        "image_filename": file.filename,
        "risk_score": risk_score,
        "confidence": confidence,
        "prediction_class": prediction_class,
        "heatmap_url": f"/heatmaps/{heatmap_filename}",
        "created_at": datetime.utcnow()
    }
    
    await db.predictions.insert_one(pred_result)
    return pred_result

@app.get("/heatmaps/{filename}")
async def get_heatmap(filename: str):
    file_path = os.path.join(OS_PATH_HEATMAPS, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(file_path)

@app.get("/download-report")
async def download_report(prediction_id: str, current_user: dict = Depends(get_current_user)):
    from fpdf import FPDF
    
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    pred_doc = await db.predictions.find_one({"id": prediction_id, "user_id": current_user["id"]})
    if not pred_doc:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Generate PDF (Simplistic)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="CardioVision AI Medical Report", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient ID: {current_user['id']}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Name: {current_user['name']}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {pred_doc['created_at'].strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Diagnosis Result", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Risk Score: {pred_doc['risk_score'] * 100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {pred_doc['confidence'] * 100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {pred_doc['prediction_class']}", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Grad-CAM Analysis Image:", ln=True)
    
    heatmap_path = os.path.join(OS_PATH_HEATMAPS, os.path.basename(pred_doc['heatmap_url']))
    if os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=10, y=pdf.get_y(), w=100)
        
    report_filename = f"report_{prediction_id}.pdf"
    report_path = os.path.join(OS_PATH_REPORTS, report_filename)
    pdf.output(report_path)
    
    return FileResponse(report_path, filename="CardioVision_Report.pdf", media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
