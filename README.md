# Customer Churn MLOps Pipeline & Deployment

This project implements an end-to-end, automated MLOps pipeline for predicting customer churn using the Telco Customer Churn dataset. It covers **Data Engineering**, **Model Engineering** (with MLflow experiment tracking), and **Containerized Deployment** (FastAPI backend + Streamlit UI) orchestrated with **DVC** and an automated 5-minute scheduler.

---

## 🏗️ Project Architecture

```text
├── code/
│   ├── datasets/
│   │   └── preprocessing.py          # Stage 1: Data cleaning, missing value handling & stratified split
│   ├── models/
│   │   ├── feature_engineering.py    # Shared feature transformations
│   │   └── training.py               # Stage 2: CatBoost training, evaluation & MLflow logging
│   └── deployment/
│       ├── api/
│       │   ├── Dockerfile            # FastAPI container build
│       │   ├── model_api.py          # FastAPI inference server & hot-reload endpoint
│       │   └── requirements.txt      # API container dependencies
│       ├── app/
│       │   ├── Dockerfile            # Streamlit container build
│       │   ├── streamlit_app.py      # Streamlit interactive web interface
│       │   └── requirements.txt      # App container dependencies
│       ├── docker-compose.yml        # Docker Compose configuration (service definitions)
│       └── scheduler.py              # Automated 5-minute DVC orchestrator & reload notifier
├── data/
│   ├── raw/
│   │   └── data_raw.csv              # Raw input data
│   └── processed/
│       ├── train.csv                 # Stratified training split
│       └── test.csv                  # Stratified testing split
├── models/
│   ├── catboost_model.cbm            # Trained CatBoost model binary
│   ├── catboost_model.pkl            # Pickled model artifact
│   └── model_features.json           # Expected model feature schema
├── dvc.yaml                          # DVC pipeline stages ('preprocess' and 'train')
├── dvc.lock                          # DVC state and dependency hashes
├── docker-compose.yml                # Root compose file for running services
├── requirements.txt                  # Full project dependencies (local & host orchestrator)
└── README.md                         # Project documentation
```

---

## 📋 Prerequisites

- **Python 3.10+** (tested on Python 3.12)
- **Docker** and **Docker Compose** installed and running
- **Git**

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Local Environment

1. Clone the repository:
   ```bash
   git clone <REPO_URL>
   cd ml_deploy_assig
   ```

2. Create and activate a virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### 2. Run the Machine Learning Pipeline (DVC)

To execute the data preprocessing and model training stages:

```bash
dvc repro
```

- **Stage 1 (`preprocess`):** Cleans invalid values in `TotalCharges`, removes non-predictive `customerID`, encodes `Churn` (0/1), and creates an 80/20 stratified split into `data/processed/train.csv` and `test.csv`.
- **Stage 2 (`train`):** Generates domain features, trains a balanced `CatBoostClassifier`, records parameters and metrics (Accuracy, F1-score, ROC-AUC) into MLflow (`sqlite:///mlflow.db`), and outputs the trained model artifacts to `models/`.

*(Optional)* To inspect experiment metrics and parameters in MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open `http://localhost:5000` in your browser.

---

### 3. Start the Docker Containers (Deployment)

Build and launch the API and Streamlit web application containers in detached mode:

```bash
docker compose up -d --build
```

Verify that both containers (`churn_api` and `churn_app`) are healthy and running:
```bash
docker compose ps
```

---

### 4. Start the Automated 5-Minute Pipeline Scheduler

The automated scheduler continuously monitors `data/raw/data_raw.csv` using DVC every 5 minutes. If changes are detected, DVC reproduces the pipeline, updates `dvc.lock`, and automatically sends a hot-reload signal to the running FastAPI container without dropping incoming traffic.

Make sure your virtual environment is active, then run:

```bash
python code/deployment/scheduler.py
```

#### Scheduler Options:
- **Default mode:** Checks every 300 seconds (5 minutes).
- **Run once (test / instant verification):**
  ```bash
  python code/deployment/scheduler.py --once
  ```
- **Custom interval (e.g. check every 60 seconds):**
  ```bash
  python code/deployment/scheduler.py --interval 60
  ```
- **Schediler should be started only with virtual environment activated!**
---

### 5. Access the Web App & API

| Component | URL | Description |
| :--- | :--- | :--- |
| **Streamlit Web App** | [http://localhost:8501](http://localhost:8501) | Interactive UI to input customer details and predict churn probability. |
| **FastAPI Interactive Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI for exploring and testing API endpoints. |
| **API Health Check** | [http://localhost:8000/](http://localhost:8000/) | Status check confirming model readiness. |
| **Model Hot-Reload** | `POST http://localhost:8000/pipeline/reload` | Reloads model weights into memory without container restart. |

#### Testing Inference via API (cURL / PowerShell)

**PowerShell:**
```powershell
$body = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    tenure = 12
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "Fiber optic"
    OnlineSecurity = "No"
    OnlineBackup = "Yes"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "Yes"
    StreamingMovies = "No"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 85.5
    TotalCharges = 1026.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 12,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "Yes",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "Yes",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 85.5,
       "TotalCharges": 1026.0
     }'
```

**Example Response:**
```json
{
  "prediction": 1,
  "churn": true,
  "churn_probability": 0.5832,
  "risk_level": "High"
}
```

