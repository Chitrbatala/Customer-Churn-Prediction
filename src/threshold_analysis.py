import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
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


# ==================================================
# 5. Define tuned models
# ==================================================

models = {

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="log2",
                    max_depth=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    ),

    "XGBoost": Pipeline(
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
    ),

    "CatBoost": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                CatBoostClassifier(
                    learning_rate=0.03,
                    l2_leaf_reg=3,
                    iterations=200,
                    depth=4,
                    random_seed=42,
                    verbose=False
                )
            )
        ]
    )
}


# ==================================================
# 6. Train models and get probabilities
# ==================================================

probabilities = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    probabilities[name] = model.predict_proba(
        X_test
    )[:, 1]


# ==================================================
# 7. Threshold Analysis
# ==================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


for name, probability in probabilities.items():

    results = []

    for threshold in thresholds:

        predictions = (
            probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })


    results_df = pd.DataFrame(results)


    print("\n")
    print("=" * 55)
    print(f"THRESHOLD ANALYSIS — {name}")
    print("=" * 55)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format
            }
        )
    )


    # ----------------------------------------------
    # Best threshold according to F1
    # ----------------------------------------------

    best_row = results_df.loc[
        results_df["F1"].idxmax()
    ]

    print("\nBest F1 Threshold:")
    print(
        f"Threshold = {best_row['Threshold']:.2f}"
    )

    print(
        f"Precision = {best_row['Precision']:.4f}"
    )

    print(
        f"Recall = {best_row['Recall']:.4f}"
    )

    print(
        f"F1 = {best_row['F1']:.4f}"
    )