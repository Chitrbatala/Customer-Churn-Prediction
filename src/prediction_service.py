from pathlib import Path
import joblib
import pandas as pd

from src.risk_engine import get_risk_level, get_retention_priority
from src.retention_engine import get_retention_recommendations
from src.shap_explainability import explain_customer

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_customer(customer):
    """
    Generate churn prediction and retention information
    for a single customer.
    """

    # Convert customer dictionary into DataFrame
    customer_df = pd.DataFrame([customer])

    # Get churn probability
    churn_probability = model.predict_proba(customer_df)[0][1]

    # Determine risk level
    risk_level = get_risk_level(churn_probability)

    # Determine retention priority
    retention_priority = get_retention_priority(risk_level)

    # Generate retention recommendations
    recommendations = get_retention_recommendations(customer)
    explanation = explain_customer(customer)

    # Final result
    result = {
        "churn_probability": round(float(churn_probability), 4),
        "churn_probability_percentage": round(
            float(churn_probability) * 100, 2
        ),
        "churn_prediction": (
            "Likely to Churn"
            if churn_probability >= 0.25
            else "Likely to Stay"
        ),
        "risk_level": risk_level,
        "retention_priority": retention_priority,
        "risk_factors": explanation["risk_increasing"],
        "protective_factors": explanation["risk_reducing"],
        "retention_recommendations": recommendations
}

    return result


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    test_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 92.0,
        "TotalCharges": 460.0
    }

    result = predict_customer(test_customer)

    print("\n========== CUSTOMER CHURN PREDICTION ==========")

    print(f"Churn Probability : {result['churn_probability']}")
    print(f"Risk Level        : {result['risk_level']}")
    print(f"Retention Priority: {result['retention_priority']}")

    print("\nRetention Recommendations:")

    for recommendation in result["retention_recommendations"]:
        print(f"- {recommendation}")