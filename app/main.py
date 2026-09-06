from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.prediction_service import predict_customer


# ==================================================
# Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ==================================================
# Create FastAPI application
# ==================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML-powered customer churn prediction and retention system",
    version="1.0.0"
)


# ==================================================
# CORS Configuration
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# Serve frontend static files
# ==================================================

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


# ==================================================
# Customer input schema
# ==================================================

class Customer(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ==================================================
# Frontend
# ==================================================

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )

@app.get("/health")
def health():
    return {"status": "healthy"}

# ==================================================
# Prediction endpoint
# ==================================================

@app.post("/predict")
def predict(customer: Customer):

    customer_data = customer.model_dump()

    result = predict_customer(customer_data)

    return result