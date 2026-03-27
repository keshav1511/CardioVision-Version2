import os
import base64
import io
import tempfile
from datetime import datetime, timedelta
from uuid import uuid4

from gradio_client import Client, handle_file
from PIL import Image

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from jose import JWTError, jwt
import sys
import time
import requests
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
# HUGGINGFACE (GRADIO CLIENT)
# ---------------------------------------------------------

import sys
import requests
HF_SPACE_ID = "keshavnayak15/cardiovision-b7-v2"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")

if HF_TOKEN:
    masked_token = HF_TOKEN[:4] + "..." + HF_TOKEN[-4:] if len(HF_TOKEN) > 8 else "****"
    print(f"DIAGNOSTIC: Found token in environment: {masked_token}")
else:
    print("DIAGNOSTIC: No HF_TOKEN or HF_API_TOKEN found in environment.")

sys.stdout.flush()

client = None

def get_gradio_client():
    global client
    if client is None:
        max_retries = 5  # Increase retries
        for i in range(max_retries):
            try:
                # Check status via API first
                headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
                api_url = f"https://huggingface.co/api/spaces/{HF_SPACE_ID}"
                resp = requests.get(api_url, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    status = resp.json().get("runtime", {}).get("stage")
                    print(f"DIAGNOSTIC: Space current status: {status}")
                    
                    if status == "SLEEPING" or status == "PAUSED":
                        print(f"DIAGNOSTIC: Space is {status}. Sending wakeup signal and waiting 30s...")
                        # Visiting the space URL wakes it up
                        requests.get(f"https://huggingface.co/spaces/{HF_SPACE_ID}", headers=headers, timeout=15)
                        time.sleep(30) # Increased wait time for cold start
                    elif status == "BUILDING" or status == "RUNNING_BUILDING":
                        print("DIAGNOSTIC: Space is building. Waiting 20s...")
                        time.sleep(20)
                
                print(f"Attempting to connect to Gradio Client (Attempt {i+1} of {max_retries})...")
                sys.stdout.flush()
                client = Client(HF_SPACE_ID, token=HF_TOKEN)
                print("Gradio Client connected successfully!")
                return client
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                sys.stdout.flush()
                if i < max_retries - 1:
                    print(f"Waiting 10s before next attempt...")
                    time.sleep(10)
                else:
                    raise HTTPException(
                        status_code=503,
                        detail=f"HuggingFace Space connection failed after {max_retries} attempts: {str(e)}"
                    )
    return client


def query_huggingface(image_bytes):
    tmp_path = None
    try:
        current_client = get_gradio_client()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        result = current_client.predict(
            handle_file(tmp_path),
            api_name="/predict"
        )

        # Robust parsing for nested lists from Gradio
        output = result
        while isinstance(output, list) and len(output) > 0:
            output = output[0]

        if not isinstance(output, dict) or "score" not in output:
            raise ValueError(f"Unexpected output format from HF Space: {result}")

        return {
            "score": output["score"],
            "heatmap": output["heatmap"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"HF request failed: {str(e)}"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


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