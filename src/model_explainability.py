from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

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
# 5. Final XGBoost Pipeline
# ==================================================

model_pipeline = Pipeline(
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
# 6. Train
# ==================================================

print("Training XGBoost...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==================================================
# 7. Get feature names
# ==================================================

feature_names = (
    model_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# ==================================================
# 8. Get feature importance
# ==================================================

importances = (
    model_pipeline
    .named_steps["model"]
    .feature_importances_
)


# ==================================================
# 9. Create importance table
# ==================================================

import pandas as pd

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ==================================================
# 10. Display top features
# ==================================================

print("\n========== TOP 20 FEATURES ==========")

print(
    importance_df
    .head(20)
    .to_string(index=False)
)