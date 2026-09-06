import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from data.data_preprocessing import (
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
# 3. Final Train / Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# 4. Preprocessing
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
# 5. Final XGBoost Model
# ==================================================

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
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
        )
    ]
)


# ==================================================
# 6. Train Final Model
# ==================================================

print("\nTraining final XGBoost model...")

final_pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==================================================
# 7. Probability Prediction
# ==================================================

y_probability = final_pipeline.predict_proba(
    X_test
)[:, 1]


# ==================================================
# 8. Final Threshold
# ==================================================

FINAL_THRESHOLD = 0.25

y_pred = (
    y_probability >= FINAL_THRESHOLD
).astype(int)


# ==================================================
# 9. Final Evaluation
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ==================================================
# 10. Display Results
# ==================================================

print("\n")
print("=" * 55)
print("             FINAL MODEL RESULTS")
print("=" * 55)

print(f"Model:      XGBoost")
print(f"Threshold:  {FINAL_THRESHOLD}")

print(f"\nAccuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1 Score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")
print(f"PR-AUC:     {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )
)


# ==================================================
# 11. Save Final Pipeline
# ==================================================

MODEL_PATH = "models/churn_model.pkl"

joblib.dump(
    final_pipeline,
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")