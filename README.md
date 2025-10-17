# 🛍️ H&M Sales Analytics

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-blue.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive end-to-end data science pipeline for H&M sales prediction with advanced analytics, model explainability, fairness auditing, and interactive dashboards.

![Dashboard Preview](http://localhost:8502/)

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Dashboard Pages](#dashboard-pages)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a complete machine learning pipeline for predicting H&M product success, featuring:

- **Data Preprocessing & EDA**: Comprehensive data cleaning and exploratory analysis
- **ML Model Training**: 10+ algorithms with hyperparameter tuning and MLflow tracking
- **Explainability**: SHAP and LIME interpretations for model transparency
- **Fairness Analysis**: Bias detection and mitigation using Fairlearn
- **API Deployment**: RESTful API with FastAPI and Docker containerization
- **Interactive Dashboard**: Real-time analytics and predictions with Streamlit

## ✨ Features

### 🧹 Data Processing
- Automated data quality checks
- Missing value analysis and handling
- Outlier detection and treatment
- Feature engineering pipeline

### 🤖 Machine Learning
- **10 Baseline Models**: Decision Tree, Logistic Regression, LightGBM, LDA, XGBoost, Naive Bayes, ANN/DNN, KNN, SVM, Random Forest
- **Hyperparameter Tuning**: RandomizedSearchCV optimization
- **Experiment Tracking**: MLflow integration for reproducibility
- **Model Comparison**: Comprehensive performance metrics

### 🔍 Model Explainability
- **SHAP (SHapley Additive exPlanations)**: Global feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Individual prediction explanations
- **Feature Impact Analysis**: Understanding model decision-making

### ⚖️ Fairness & Bias
- **Fairlearn Integration**: Demographic parity and equalized odds analysis
- **Bias Detection**: Cross-group performance disparities
- **Mitigation Strategies**: Pre-processing, in-processing, and post-processing techniques

### 🚀 Deployment
- **FastAPI Backend**: RESTful API with automatic documentation
- **Docker Containerization**: Consistent deployment across environments
- **Health Monitoring**: Built-in health checks and logging
- **Batch Predictions**: Support for single and batch inference

### 📊 Interactive Dashboard
- **Real-time Analytics**: Dynamic filtering and visualization
- **Predictive Interface**: User-friendly prediction tool
- **Business Insights**: KPIs, trends, and product segmentation
- **Export Capabilities**: Download filtered data and reports

## 🏗️ Project Architecture

```
┌─────────────────┐
│   Data Layer    │
│  (CSV/Database) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Pipeline  │
│ (Cleaning, EDA) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  ML Pipeline    │─────▶│   MLflow     │
│ (Training, Eval)│      │  Tracking    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  Explainability │
│  (SHAP, LIME)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fairness Audit │
│   (Fairlearn)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Deployment    │─────▶│   Docker     │
│   (FastAPI)     │      │  Container   │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│    Dashboard    │
│   (Streamlit)   │
└─────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Docker (optional, for containerization)

### Clone Repository
```bash
git clone https://github.com/yourusername/hm-sales-analytics.git
cd hm-sales-analytics
```

### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
streamlit==1.28.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
mlflow==2.8.1
shap==0.43.0
lime==0.2.0.1
fairlearn==0.9.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
lightgbm==4.1.0
xgboost==2.0.2
catboost==1.2.2
python-multipart==0.0.6
```

## 💻 Usage

### Run Streamlit Dashboard
```bash
streamlit run dashboard.py
```
Access at: `http://localhost:8501`

### Run FastAPI Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation: `http://localhost:8000/docs`

### Docker Deployment
```bash
# Build Docker image
docker build -t hm-sales-api:latest .

# Run container
docker run -d -p 8000:8000 --name hm-api hm-sales-api:latest

# Using Docker Compose
docker-compose up -d
```

### MLflow Tracking
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```
Access at: `http://localhost:5000`

## 📁 Project Structure

```
hm-sales-analytics/
│
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Cleaned datasets
│   └── sample/                 # Sample data for testing
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_fairness.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data cleaning utilities
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # ML training pipeline
│   ├── explainability.py       # SHAP/LIME implementations
│   └── fairness.py             # Fairlearn utilities
│
├── models/
│   ├── best_model.pkl          # Trained model
│   ├── preprocessor.pkl        # Data preprocessor
│   └── model_metadata.json     # Model info
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── schemas.py              # Pydantic models
│   └── utils.py                # Helper functions
│
├── mlruns/                     # MLflow experiments
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_api.py
│
├── assets/
│   └── dashboard_preview.png
│
├── dashboard.py                # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── .gitignore
├── LICENSE
└── README.md
```

## 🛠️ Technologies Used

### Core ML & Data Science
- **Python 3.9+**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **LightGBM, XGBoost, CatBoost**: Gradient boosting frameworks

### Visualization
- **Matplotlib & Seaborn**: Statistical visualizations
- **Plotly**: Interactive dashboards and charts

### ML Operations
- **MLflow**: Experiment tracking and model registry
- **SHAP**: Model explainability
- **LIME**: Local interpretations
- **Fairlearn**: Fairness assessment and mitigation

### Deployment
- **Streamlit**: Interactive web dashboard
- **FastAPI**: RESTful API framework
- **Uvicorn**: ASGI server
- **Docker**: Containerization
- **Pydantic**: Data validation

## 📊 Model Performance

### Best Performing Models

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **KNN** | **94.92%** | **96.10%** | **96.28%** | **96.19%** | 6.13s |
| ANN/DNN | 94.13% | 97.42% | 93.66% | 95.50% | 18.92s |
| LDA | 91.33% | 91.03% | 91.67% | 91.35% | 0.13s |
| Random Forest | 100%* | 100%* | 100%* | 100%* | 3.34s |

*Note: 100% accuracy models show signs of overfitting*

### Feature Importance (SHAP)
1. **Price** (24%)
2. **Product Type** (18%)
3. **Color** (15%)
4. **Season** (12%)
5. **Customer Age** (9%)

### Fairness Metrics
- **Demographic Parity Difference**: 0.16 → 0.04 (after mitigation)
- **Accuracy Disparity**: 7.9% → 2.1% (after mitigation)
- **Equalized Odds**: Achieved across gender groups

## 🔌 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "price": 49.99,
  "product_type": "Dress",
  "color": "Blue",
  "season": "Summer",
  "customer_age": 28,
  "brand_score": 8.5,
  "discount": 0.15,
  "stock_level": 150,
  "previous_purchases": 5,
  "review_score": 4.2,
  "size": "M",
  "material": "Cotton",
  "gender": "Women"
}
```
Response:
```json
{
  "prediction": 1,
  "probability": 0.92,
  "confidence": "High"
}
```

#### Batch Predictions
```http
POST /predict_batch
Content-Type: application/json

{
  "items": [...]
}
```

## 📱 Dashboard Pages

### 1. 🏠 Introduction
- Project overview and objectives
- Technology stack information
- Pipeline architecture visualization

### 2. 🧹 Data Cleaning
- Upload and preview datasets
- Missing value analysis
- Data quality reports
- Export cleaned data

### 3. 📊 EDA & Statistical Analysis
- Univariate distributions
- Bivariate relationships
- Correlation heatmaps
- Statistical summaries

### 4. 🤖 ML Modeling & Tracking
- Baseline model comparison
- Hyperparameter tuning results
- MLflow experiment tracking
- Model selection criteria

### 5. 🔍 Explainability & Fairness
- SHAP global feature importance
- LIME local explanations
- Fairness audit results
- Bias mitigation strategies

### 6. 🚀 API Deployment
- API documentation
- Docker configuration
- Deployment instructions
- Testing interface

### 7. 📈 Dashboard & Predictions
- Real-time KPI metrics
- Interactive prediction tool
- Sales analytics
- Product insights

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guide
- Add docstrings to functions
- Write unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Your Name**
- Email: your.email@example.com
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)
- Portfolio: [yourportfolio.com](https://yourportfolio.com)

**Project Link**: [https://github.com/yourusername/hm-sales-analytics](https://github.com/yourusername/hm-sales-analytics)

## 🙏 Acknowledgments

- H&M Group for domain inspiration
- Scikit-learn community for ML tools
- MLflow team for experiment tracking
- SHAP and LIME developers for explainability
- Fairlearn team for fairness tools
- Streamlit for dashboard framework

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
