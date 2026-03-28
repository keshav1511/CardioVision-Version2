import os
import base64
import io
import tempfile
from datetime import datetime, timedelta
from uuid import uuid4

from PIL import Image

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from jose import JWTError, jwt
import anthropic

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
    allow_credentials=False,
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


from backend.gradcam import predict as local_predict, generate_gradcam


# ---------------------------------------------------------
# ANTHROPIC (RETINAL IMAGE VALIDATION)
# ---------------------------------------------------------

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def validate_retinal_image(image_bytes: bytes) -> bool:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        message = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Is this a retinal fundus image (an eye scan/photograph of the back of the eye)? Reply with only YES or NO."
                        }
                    ],
                }
            ],
        )

        answer = message.content[0].text.strip().upper()
        return answer == "YES"

    except Exception:
        return True  # fail open — don't block if Claude is unavailable


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

@app.api_route("/", methods=["GET", "HEAD"])
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

    # Validate retinal image using Claude
    if not validate_retinal_image(contents):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid retinal fundus image."
        )

    risk_score = local_predict(contents)
    prediction_class = "High Risk" if risk_score > 0.6 else "Low Risk"

    heatmap_filename = f"{uuid4()}.jpg"
    heatmap_path = os.path.join(OS_PATH_HEATMAPS, heatmap_filename)

    # Generate Grad-CAM heatmap locally
    generate_gradcam(contents, heatmap_path)

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
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Calculate prediction sequence number for this user
    count = await db.predictions.count_documents({
        "user_id": current_user["id"],
        "created_at": {"$lte": pred["created_at"]}
    })

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "CardioVision Medical Report", ln=True, align='C')
    pdf.ln(10)

    # Patient Info
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Patient Name: {current_user['name']}", ln=True)
    pdf.cell(200, 10, f"Date: {pred['created_at'].strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, f"Report ID: {prediction_id}", ln=True)
    pdf.ln(5)

    # Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Analysis Results", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Risk Score: {pred['risk_score']*100:.2f}%", ln=True)
    pdf.cell(200, 10, f"Classification: {pred['prediction_class']}", ln=True)
    pdf.ln(10)

    # Heatmap Image
    try:
        # Resolve local path for heatmap
        # Format of heatmap_url is /heatmaps/filename.jpg
        heatmap_filename = pred["heatmap_url"].split("/")[-1]
        heatmap_path = os.path.join(OS_PATH_HEATMAPS, heatmap_filename)
        
        if os.path.exists(heatmap_path):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Risk Localization (Heatmap):", ln=True)
            pdf.image(heatmap_path, x=10, y=pdf.get_y(), w=100)
            pdf.ln(105) # Space for image
    except Exception as e:
        print(f"Failed to add image to PDF: {e}")
        pdf.cell(200, 10, "(Image unavailable in report)", ln=True)

    # Footer
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "This report is generated by CardioVision AI and should be reviewed by a medical professional.", 0, 0, 'C')

    # File naming: patientname_count.pdf
    safe_name = current_user['name'].replace(" ", "_").lower()
    filename = f"{safe_name}_{count}.pdf"
    
    report_file = os.path.join(OS_PATH_REPORTS, f"{prediction_id}.pdf")
    pdf.output(report_file)

    return FileResponse(
        report_file, 
        filename=filename,
        media_type="application/pdf"
    )