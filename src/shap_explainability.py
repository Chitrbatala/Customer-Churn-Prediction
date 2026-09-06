import shap
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.explanation_engine import get_contributing_features
from xgboost import XGBClassifier

from src.data.data_preprocessing import (
    load_data,
    clean_data,
    prepare_features
)

# ==================================================
# 1. Load and clean data
# ==================================================

df = load_data()
df = clean_data(df)

X, y = prepare_features(df)


# ==================================================
# 2. Feature definitions
# ==================================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ==================================================
# 3. Train / Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# 4. Preprocessor
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ==================================================
# 5. XGBoost model
# ==================================================

model = XGBClassifier(
    subsample=0.9,
    n_estimators=200,
    min_child_weight=5,
    max_depth=4,
    learning_rate=0.03,
    gamma=0,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# ==================================================
# 6. Preprocess training data
# ==================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ==================================================
# 7. Train XGBoost
# ==================================================

print("Training XGBoost...")

model.fit(
    X_train_processed,
    y_train
)

print("Training completed!")


# ==================================================
# 8. Create SHAP explainer
# ==================================================

explainer = shap.TreeExplainer(model)

# ==================================================
# 9. Create reusable SHAP explanation function
# ==================================================

def explain_customer(customer):

    # Convert customer dictionary into DataFrame
    customer_df = pd.DataFrame([customer])

    # Preprocess customer
    customer_processed = preprocessor.transform(
        customer_df
    )

    # Calculate SHAP values
    shap_values = explainer.shap_values(
        customer_processed
    )

    # Get feature names
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Create explanation table
    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values[0],
        "Feature_Value": customer_processed[0],
    })

    # Absolute SHAP value
    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )

    # Keep active features
    active_explanation = explanation[
        explanation["Feature_Value"] != 0
    ].copy()

    # Sort by importance
    active_explanation = (
        active_explanation
        .sort_values(
            by="Absolute_SHAP",
            ascending=False
        )
    )

    # Generate human-readable explanation
    explanation_result = get_contributing_features(
        explanation_df=active_explanation,
        top_n=5
    )

    return explanation_result


# ==================================================
# 10. Test SHAP explanation
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