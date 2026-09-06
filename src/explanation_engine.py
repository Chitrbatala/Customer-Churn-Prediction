# ==================================================
# Feature Name Formatting
# ==================================================

def clean_feature_name(feature_name):

    # Remove preprocessing prefixes
    feature_name = feature_name.replace(
        "cat__",
        ""
    )

    feature_name = feature_name.replace(
        "num__",
        ""
    )


    # --------------------------------------------------
    # Explicit business-friendly mappings
    # --------------------------------------------------

    feature_map = {

        "Contract_Month-to-month":
            "Month-to-month contract",

        "Contract_One year":
            "One-year contract",

        "Contract_Two year":
            "Two-year contract",

        "InternetService_Fiber optic":
            "Fiber optic internet service",

        "InternetService_DSL":
            "DSL internet service",

        "InternetService_No":
            "No internet service",

        "TechSupport_No":
            "No technical support",

        "TechSupport_Yes":
            "Technical support",

        "OnlineSecurity_No":
            "No online security",

        "OnlineSecurity_Yes":
            "Online security",

        "OnlineBackup_No":
            "No online backup",

        "OnlineBackup_Yes":
            "Online backup",

        "DeviceProtection_No":
            "No device protection",

        "DeviceProtection_Yes":
            "Device protection",

        "PaymentMethod_Electronic check":
            "Electronic check payment",

        "PaymentMethod_Mailed check":
            "Mailed check payment",

        "PaymentMethod_Bank transfer (automatic)":
            "Automatic bank transfer",

        "PaymentMethod_Credit card (automatic)":
            "Automatic credit card payment",

        "PaperlessBilling_Yes":
            "Paperless billing",

        "PaperlessBilling_No":
            "Paper billing",

        "PhoneService_Yes":
            "Phone service",

        "PhoneService_No":
            "No phone service",

        "MultipleLines_Yes":
            "Multiple phone lines",

        "MultipleLines_No":
            "Single phone line",

        "StreamingTV_Yes":
            "Streaming TV",

        "StreamingTV_No":
            "No streaming TV",

        "StreamingMovies_Yes":
            "Streaming movies",

        "StreamingMovies_No":
            "No streaming movies",

        "Partner_Yes":
            "Has a partner",

        "Partner_No":
            "No partner",

        "Dependents_Yes":
            "Has dependents",

        "Dependents_No":
            "No dependents",

        "gender_Male":
            "Male customer",

        "gender_Female":
            "Female customer",

        "SeniorCitizen":
            "Senior citizen",

        "tenure":
            "Customer tenure",

        "MonthlyCharges":
            "Monthly charges",

        "TotalCharges":
            "Total charges"
    }


    if feature_name in feature_map:

        return feature_map[feature_name]


    # Fallback formatting
    return feature_name.replace(
        "_",
        " "
    )


# ==================================================
# Get Contributing Features
# ==================================================

def get_contributing_features(
    explanation_df,
    top_n=5
):

    # --------------------------------------------------
    # Risk Increasing Features
    # --------------------------------------------------

    positive = explanation_df[
        explanation_df["SHAP_Value"] > 0
    ].sort_values(
        by="SHAP_Value",
        ascending=False
    ).head(top_n)


    # --------------------------------------------------
    # Risk Reducing Features
    # --------------------------------------------------

    negative = explanation_df[
        explanation_df["SHAP_Value"] < 0
    ].sort_values(
        by="SHAP_Value"
    ).head(top_n)


    # --------------------------------------------------
    # Format Risk Increasing Features
    # --------------------------------------------------

    positive_features = []

    for _, row in positive.iterrows():

        positive_features.append({

            "feature":
                clean_feature_name(
                    row["Feature"]
                ),

            "shap_value":
                float(row["SHAP_Value"]),

            "effect":
                "increases churn risk"

        })


    # --------------------------------------------------
    # Format Risk Reducing Features
    # --------------------------------------------------

    negative_features = []

    for _, row in negative.iterrows():

        negative_features.append({

            "feature":
                clean_feature_name(
                    row["Feature"]
                ),

            "shap_value":
                float(row["SHAP_Value"]),

            "effect":
                "decreases churn risk"

        })


    # --------------------------------------------------
    # Final Explanation
    # --------------------------------------------------

    return {

        "risk_increasing":
            positive_features,

        "risk_reducing":
            negative_features

    }