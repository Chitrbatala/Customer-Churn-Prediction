# Customer Churn Prediction & Retention System

An end-to-end Machine Learning application that predicts customer churn probability, explains the factors behind the prediction, and provides retention recommendations.

## 🚀 Live Demo

[Customer Churn Prediction](https://customer-churn-prediction-9tmf.onrender.com)

## 📌 Project Overview

This project uses the IBM Telco Customer Churn dataset to identify customers who are likely to churn.

The system provides:

- Churn probability
- Churn prediction
- Customer risk level
- Retention priority
- SHAP-based risk factors
- Retention recommendations

## 📊 Dataset

- 7,043 customers
- 21 features
- Target: `Churn`

Some important features include:

- Tenure
- Contract
- Internet Service
- Monthly Charges
- Total Charges
- Payment Method
- Technical Support
- Online Security

### Key Findings

- Month-to-month customers have significantly higher churn.
- New customers show higher churn rates.
- Customers with higher monthly charges tend to churn more.
- Electronic check users show higher churn.
- Customers without technical support or online security show higher churn.

## 🤖 Machine Learning

Several models were compared:

| Model | ROC-AUC | F1 |
|---|---:|---:|
| Logistic Regression | 0.8421 | 0.6040 |
| Random Forest | 0.8195 | 0.5397 |
| XGBoost | **0.8468** | **0.6355** |
| CatBoost | 0.8456 | 0.5775 |

The final system uses **XGBoost** with a churn classification threshold of `0.25` to prioritize recall for retention purposes.

### Final Model Performance

- Accuracy: **75.09%**
- Precision: **51.95%**
- Recall: **81.82%**
- F1-score: **63.55%**
- ROC-AUC: **84.68%**
- PR-AUC: **66.06%**

## 🔍 Explainable AI

**SHAP** is used to explain individual predictions.

The application identifies:

- Factors increasing churn risk
- Factors reducing churn risk

This helps translate model predictions into business-friendly insights.

## 💼 Retention Engine

Based on customer characteristics and churn risk, the system provides recommendations such as:

- Encourage longer-term contracts
- Offer technical support
- Consider online security/backup packages
- Review high monthly charges
- Encourage automatic payment methods
- Provide early-lifecycle retention support

These recommendations are rule-based business suggestions, not causal predictions.

## 🏗️ Tech Stack

**Machine Learning:**  
Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost

**Explainability:**  
SHAP

**Backend:**  
FastAPI, Uvicorn, Pydantic

**Frontend:**  
HTML, CSS, JavaScript

**Deployment:**  
Docker, Render

**Version Control:**  
Git, GitHub

## 📁 Project Structure

```text
Customer-Churn-Prediction/
├── app/
├── data/
├── frontend/
├── models/
├── notebooks/
├── src/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md

🐳 Run Locally with Docker
docker build -t customer-churn-prediction .
docker run -d -p 8000:8000 --name customer-churn-app customer-churn-prediction

Open:

http://localhost:8000

Health check:

http://localhost:8000/health

👨‍💻 Author

Chitr Batala

B.Tech — Computer Science / Artificial Intelligence & Machine Learning

Interested in Data Science, Machine Learning, Data Analytics, and AI/ML Engineering.