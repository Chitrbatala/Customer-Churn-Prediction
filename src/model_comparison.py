from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from data.data_preprocessing import (
    load_data,
    clean_data,
    prepare_features
)


# --------------------------------------------------
# 1. Load and clean data
# --------------------------------------------------

df = load_data()
df = clean_data(df)

X, y = prepare_features(df)


# --------------------------------------------------
# 2. Feature types
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Train / Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Preprocessor
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Random Forest Pipeline
# --------------------------------------------------

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# --------------------------------------------------
# 6. Train
# --------------------------------------------------

rf_pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 7. Predictions
# --------------------------------------------------

y_pred = rf_pipeline.predict(X_test)

y_probability = rf_pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print("\n========== RANDOM FOREST ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

# --------------------------------------------------
# 9. XGBoost Pipeline
# --------------------------------------------------

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)


# --------------------------------------------------
# 10. Train XGBoost
# --------------------------------------------------

xgb_pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 11. XGBoost Predictions
# --------------------------------------------------

xgb_pred = xgb_pipeline.predict(X_test)

xgb_probability = xgb_pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# 12. XGBoost Evaluation
# --------------------------------------------------

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

xgb_precision = precision_score(
    y_test,
    xgb_pred
)

xgb_recall = recall_score(
    y_test,
    xgb_pred
)

xgb_f1 = f1_score(
    y_test,
    xgb_pred
)

xgb_roc_auc = roc_auc_score(
    y_test,
    xgb_probability
)

xgb_pr_auc = average_precision_score(
    y_test,
    xgb_probability
)


print("\n========== XGBOOST ==========")

print(f"Accuracy:  {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall:    {xgb_recall:.4f}")
print(f"F1 Score:  {xgb_f1:.4f}")
print(f"ROC-AUC:   {xgb_roc_auc:.4f}")
print(f"PR-AUC:    {xgb_pr_auc:.4f}")

# --------------------------------------------------
# 13. CatBoost Pipeline
# --------------------------------------------------

catboost_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False
            )
        )
    ]
)


# --------------------------------------------------
# 14. Train CatBoost
# --------------------------------------------------

catboost_pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 15. CatBoost Predictions
# --------------------------------------------------

catboost_pred = catboost_pipeline.predict(X_test)

catboost_probability = catboost_pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# 16. CatBoost Evaluation
# --------------------------------------------------

cat_accuracy = accuracy_score(
    y_test,
    catboost_pred
)

cat_precision = precision_score(
    y_test,
    catboost_pred
)

cat_recall = recall_score(
    y_test,
    catboost_pred
)

cat_f1 = f1_score(
    y_test,
    catboost_pred
)

cat_roc_auc = roc_auc_score(
    y_test,
    catboost_probability
)

cat_pr_auc = average_precision_score(
    y_test,
    catboost_probability
)


print("\n========== CATBOOST ==========")

print(f"Accuracy:  {cat_accuracy:.4f}")
print(f"Precision: {cat_precision:.4f}")
print(f"Recall:    {cat_recall:.4f}")
print(f"F1 Score:  {cat_f1:.4f}")
print(f"ROC-AUC:   {cat_roc_auc:.4f}")
print(f"PR-AUC:    {cat_pr_auc:.4f}")