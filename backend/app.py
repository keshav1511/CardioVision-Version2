import os
import requests
import base64
from datetime import datetime, timedelta
from uuid import uuid4
from gradio_client import Client

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
# HUGGINGFACE CONFIG
# ---------------------------------------------------------

HF_API_URL = "https://keshavnayak15-cardiovision-b7-v2.hf.space/api/predict"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}


from gradio_client import Client
import base64

client = Client("keshavnayak15/cardiovision-b7-v2")

def query_huggingface(image_bytes):
    try:
        # Convert to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(image_bytes)
            temp_path = f.name

        result = client.predict(
            temp_path,
            api_name="/predict"
        )

        # result = [ {score, heatmap} ]
        output = result[0]

        return {
            "score": output["score"],
            "heatmap": output["heatmap"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"HF request failed: {str(e)}"
        )


# ---------------------------------------------------------
# STARTUP / SHUTDOWN
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
# PREDICT
# ---------------------------------------------------------

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    db = get_database()

    contents = await file.read()

    result = query_huggingface(contents)

    try:
        risk_score = result["score"]
        heatmap_base64 = result["heatmap"]
    except:
        raise HTTPException(status_code=500, detail="Invalid HF response")

    prediction_class = "High Risk" if risk_score > 0.6 else "Low Risk"

    # Save heatmap locally
    heatmap_filename = f"{uuid4()}.jpg"
    heatmap_path = os.path.join(OS_PATH_HEATMAPS, heatmap_filename)

    with open(heatmap_path, "wb") as f:
        f.write(base64.b64decode(heatmap_base64))

    heatmap_url = f"/heatmaps/{heatmap_filename}"

    pred_doc = {
        "id": str(uuid4()),
        "user_id": current_user["id"],
        "image_filename": file.filename,
        "risk_score": risk_score,
        "confidence": risk_score,
        "prediction_class": prediction_class,
        "heatmap_url": heatmap_url,
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

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Heatmap not found")

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
    pdf.cell(200, 10, "CardioVision Medical Report", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Patient: {current_user['name']}", ln=True)
    pdf.cell(200, 10, f"Risk Score: {pred['risk_score']*100:.2f}%", ln=True)
    pdf.cell(200, 10, f"Prediction: {pred['prediction_class']}", ln=True)

    report_file = os.path.join(OS_PATH_REPORTS, f"{prediction_id}.pdf")
    pdf.output(report_file)

    return FileResponse(report_file, filename="CardioVision_Report.pdf")