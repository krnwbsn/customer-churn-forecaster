# Customer Churn Forecaster

## Data Source

Data used in this project is from the [WA-FnUsec Telco Customer Churn](https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn) dataset on Kaggle .

An end-to-end Python project for analyzing, visualizing, and predicting customer churn in a telecom dataset. It includes:

- **Exploratory Data Analysis & Visualization** (`src/plots.py`)  
- **Data Cleaning & Feature Engineering** (`src/data.py`, `src/features.py`)  
- **Classification Pipeline** with Logistic Regression (or LightGBM) (`src/model.py`)  
- **Evaluation & Cross-Validation** (`src/evaluate.py`)  
- **Hyperparameter Tuning** via RandomizedSearchCV (`src/tuning.py`)  
- **Model Interpretation** with SHAP (`src/interpret.py`)  
- **Artifact Management** (train & dump) (`src/serve.py`)  
- **Batch Inference** for all customers (`bulk_inference.py`)  
- **Single-Record Inference** (`inference.py`)  
- **REST API** for on-demand scoring using FastAPI (`src/api.py`)  

---

## Project Structure
```
customer-churn-forecaster/
├── data/
│ └── telco-customer-churn.csv # Raw dataset
├── outputs/
│ ├── *.png # EDA & SHAP visualizations
│ ├── churn_model_artifacts.pkl # Trained pipeline & transformer
│ └── churn_scores.csv # Batch churn probabilities
└── src/
  ├── init.py
  ├── api.py 
  ├── bulk_inference.py 
  ├── inference.py 
  ├── serve.py 
  ├── interpret.py 
  ├── tuning.py 
  ├── evaluate.py 
  ├── model.py 
  ├── features.py 
  ├── data.py
  ├── main.py
  ├── plots.py
  ├── utils.py
  └── config.py
```

---

## Installation

1. **Clone** the repository:
  ```bash
  git clone <repo-url> customer-churn-forecaster
  cd customer-churn-forecaster
  ```

2. **Create & activate** a Python 3.10+ virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

3. **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. **Run EDA, Modeling & Inference**
All steps—from data loading through batch inference—are orchestrated in `src/main.py`. To execute end-to-end:
  ```bash
  python -m src.main
  ```

Results will be saved under `outputs/`:
- EDA plots (`*.png`)
- Model metrics printed to console
- SHAP visualizations (`shap_summary_bar.png`, `shap_summary_dot.png`)
- Model artifact (`churn_model_artifacts.pkl`)
- Batch scores (`churn_scores.csv`)

2. **Start REST API**
Serve the model for on-demand scoring via FastAPI:
  ```bash
  uvicorn src.api:app --reload --port 8000
  ```

- Swagger UI at http://localhost:8000/docs
Submit a POST to `/predict` with a JSON payload matching the `Customer` schema.

## License
This project is licensed under MIT License.