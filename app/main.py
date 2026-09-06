from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.prediction_service import predict_customer

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
# Health check
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# ==================================================
# Prediction endpoint
# ==================================================

@app.post("/predict")
def predict(customer: Customer):

    customer_data = customer.model_dump()

    result = predict_customer(customer_data)

    return result