import joblib
import shap
import pandas as pd

from pathlib import Path

from src.explanation_engine import get_contributing_features


# ==================================================
# 1. Locate saved model
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.pkl"
)


# ==================================================
# 2. Load trained pipeline
# ==================================================

pipeline = joblib.load(MODEL_PATH)


# ==================================================
# 3. Extract preprocessing and model components
# ==================================================

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# ==================================================
# 4. Create SHAP explainer
# ==================================================

explainer = shap.TreeExplainer(model)


# ==================================================
# 5. Reusable SHAP explanation function
# ==================================================

def explain_customer(customer):

    # ----------------------------------------------
    # Convert customer dictionary into DataFrame
    # ----------------------------------------------

    customer_df = pd.DataFrame([customer])


    # ----------------------------------------------
    # Preprocess customer
    # ----------------------------------------------

    customer_processed = preprocessor.transform(
        customer_df
    )


    # ----------------------------------------------
    # Calculate SHAP values
    # ----------------------------------------------

    shap_values = explainer.shap_values(
        customer_processed
    )


    # ----------------------------------------------
    # Get feature names
    # ----------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    # ----------------------------------------------
    # Create explanation table
    # ----------------------------------------------

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values[0],
        "Feature_Value": customer_processed[0],
    })


    # ----------------------------------------------
    # Absolute SHAP value
    # ----------------------------------------------

    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )


    # ----------------------------------------------
    # Keep active features
    # ----------------------------------------------

    active_explanation = explanation[
        explanation["Feature_Value"] != 0
    ].copy()


    # ----------------------------------------------
    # Sort by importance
    # ----------------------------------------------

    active_explanation = (
        active_explanation
        .sort_values(
            by="Absolute_SHAP",
            ascending=False
        )
    )


    # ----------------------------------------------
    # Generate human-readable explanation
    # ----------------------------------------------

    explanation_result = get_contributing_features(
        explanation_df=active_explanation,
        top_n=5
    )


    return explanation_result


# ==================================================
# 6. Test SHAP explanation
# ==================================================

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


    explanation_result = explain_customer(
        test_customer
    )


    print(
        "\n========== HUMAN-READABLE EXPLANATION =========="
    )


    print("\nRisk Increasing Factors:")

    for item in explanation_result["risk_increasing"]:

        print(
            f"- {item['feature']} "
            f"→ {item['effect']}"
        )


    print("\nRisk Reducing Factors:")

    for item in explanation_result["risk_reducing"]:

        print(
            f"- {item['feature']} "
            f"→ {item['effect']}"
        )