# 🔥 Forest Fire Prediction — End-to-End ML Application

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Model-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![AWS](https://img.shields.io/badge/AWS-Elastic_Beanstalk-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/elasticbeanstalk/)

> **Predict the Fire Weather Index (FWI) using real-time weather and environmental data.**
>
> A complete, production-ready machine learning web application — from exploratory data analysis to model training to cloud deployment.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Input Features Reference](#-input-features-reference)
- [Deployment](#-deployment)
  - [AWS Elastic Beanstalk](#aws-elastic-beanstalk)
  - [Heroku](#heroku)
- [Notebooks](#-notebooks)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🌍 Overview

Forest fires are among the most devastating natural disasters, destroying ecosystems, property, and lives. Early prediction and risk assessment can help authorities and communities take preventive action.

This project builds a **Fire Weather Index (FWI) prediction system** using historical data from the **Algerian Forest Fires dataset**. A **Ridge Regression** model is trained on weather and environmental features, wrapped in a Flask web application, and deployed to the cloud — demonstrating a full end-to-end ML engineering workflow.

### What is FWI?

The **Fire Weather Index (FWI)** is a numeric rating of fire intensity. It combines weather conditions and fuel moisture levels to estimate fire behavior. A higher FWI indicates greater fire danger.

---

## 🎥 Demo

1. **Home Page** — A simple welcome page at the root URL.
2. **Prediction Page** — Enter weather parameters and get an instant FWI prediction.

```
Home Page  ──▶  /
Prediction ──▶  /predictdata
```

Enter values like Temperature, Relative Humidity, Wind Speed, and other fire indices to receive a predicted FWI score.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Prediction** | Ridge Regression model predicts FWI from 9 weather/fire parameters |
| 🌐 **Web Interface** | Clean Flask-based form for real-time predictions |
| 📊 **EDA Notebook** | Detailed exploratory data analysis and feature engineering |
| 📓 **Training Notebook** | Complete model selection, training, and evaluation pipeline |
| 📦 **Pre-trained Models** | Serialized model and scaler ready for instant deployment |
| ☁️ **Cloud-Ready** | Configured for AWS Elastic Beanstalk and Heroku deployment |
| 🔄 **StandardScaler** | Input normalization for consistent predictions |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.8+ |
| **Web Framework** | Flask |
| **ML Library** | scikit-learn (Ridge Regression) |
| **Data Processing** | pandas, NumPy |
| **Web Server** | Gunicorn (production) |
| **Cloud Deployment** | AWS Elastic Beanstalk |
| **Alternate Deployment** | Heroku |

---

## 📁 Project Structure

```
forestfire/
│
├── .ebextensions/
│   └── python.config            # AWS Elastic Beanstalk WSGI configuration
│
├── dataset/
│   └── Algerian_forest_fires_cleaned_dataset.csv   # Cleaned dataset (244 rows)
│
├── models/
│   ├── ridge.pkl                # Trained Ridge Regression model
│   └── scaler.pkl               # Fitted StandardScaler
│
├── notebooks/
│   ├── 2.0-EDA And FE Algerian Forest Fires.ipynb  # Exploratory Data Analysis
│   └── 3.0-Model Training.ipynb                    # Model Training & Evaluation
│
├── templates/
│   ├── index.html               # Home / welcome page
│   └── home.html                # FWI prediction form & results
│
├── application.py               # Flask application entry point
├── Procfile                     # Gunicorn process configuration
├── requirements.txt             # Python dependencies
└── README.md                    # You are here!
```

---

## 📊 Dataset

**Source:** [Algerian Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++)

| Property | Details |
|----------|---------|
| **File** | `dataset/Algerian_forest_fires_cleaned_dataset.csv` |
| **Records** | 244 observations |
| **Features** | 14 attributes + 1 target |
| **Time Period** | June to September 2012 |
| **Regions** | Two regions in Algeria (Bejaia & Sidi Bel-Abbes) |
| **Target Variable** | FWI (Fire Weather Index) |

### Feature Breakdown

| Feature | Description | Type |
|---------|-------------|------|
| `day`, `month`, `year` | Date of observation | Temporal |
| `Temperature` | Noon temperature (°C) | Weather |
| `RH` | Relative Humidity (%) | Weather |
| `Ws` | Wind Speed (km/h) | Weather |
| `Rain` | Total rain in a day (mm) | Weather |
| `FFMC` | Fine Fuel Moisture Code | FWI Component |
| `DMC` | Duff Moisture Code | FWI Component |
| `DC` | Drought Code | FWI Component |
| `ISI` | Initial Spread Index | FWI Component |
| `BUI` | Buildup Index | FWI Component |
| `FWI` | Fire Weather Index (**Target**) | Output |
| `Classes` | Fire / Not Fire classification | Label |
| `Region` | Region identifier (0 or 1) | Category |

---

## 🧠 ML Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Raw Data   │───▶│   EDA & FE   │───▶│   Scaling   │───▶│   Ridge    │
│  (CSV)      │    │  (Notebook)  │    │ (Standard   │    │ Regression │
│             │    │              │    │  Scaler)    │    │  Model     │
└─────────────┘    └──────────────┘    └─────────────┘    └─────┬──────┘
                                                                │
                                                                ▼
                                                        ┌──────────────┐
                                                        │  FWI Score   │
                                                        │ (Prediction) │
                                                        └──────────────┘
```

**Model:** Ridge Regression (L2-regularized linear regression)
- Handles multicollinearity between weather features
- Prevents overfitting on the small dataset (244 samples)
- Fast inference suitable for real-time web predictions

**Preprocessing:** StandardScaler
- Zero mean, unit variance normalization
- Ensures all features contribute equally regardless of scale

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8** or higher
- **pip** (Python package manager)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/aaadityasngh/forestfire-main_complete_end_to_end_implementation_deployment.git
   cd forestfire-main_complete_end_to_end_implementation_deployment
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

```bash
python application.py
```

The application starts on **http://0.0.0.0:5000/**

Open your browser and navigate to:
- `http://localhost:5000/` — Home page
- `http://localhost:5000/predictdata` — Prediction form

---

## 💡 Usage

1. Navigate to `/predictdata` in your browser.
2. Fill in the weather and fire index parameters:

   | Field | Example Value | Description |
   |-------|--------------|-------------|
   | Temperature | `30` | Noon temperature in °C |
   | RH | `55` | Relative humidity in % |
   | Ws | `18` | Wind speed in km/h |
   | Rain | `0.0` | Daily rainfall in mm |
   | FFMC | `85.0` | Fine Fuel Moisture Code (0–101) |
   | DMC | `25.0` | Duff Moisture Code |
   | ISI | `5.0` | Initial Spread Index |
   | Classes | `1` | Fire class (1 = fire, 0 = not fire) |
   | Region | `0` | Region identifier (0 or 1) |

3. Click **Predict** to get the FWI score.

---

## 📡 API Reference

### `GET /`

Returns the home / welcome page.

**Response:** HTML page

---

### `GET /predictdata`

Returns the prediction form.

**Response:** HTML page with input form

---

### `POST /predictdata`

Accepts weather parameters and returns the FWI prediction.

**Request Body** (`application/x-www-form-urlencoded`):

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `Temperature` | float | ✅ | Noon temperature (°C) |
| `RH` | float | ✅ | Relative Humidity (%) |
| `Ws` | float | ✅ | Wind Speed (km/h) |
| `Rain` | float | ✅ | Total rain (mm) |
| `FFMC` | float | ✅ | Fine Fuel Moisture Code |
| `DMC` | float | ✅ | Duff Moisture Code |
| `ISI` | float | ✅ | Initial Spread Index |
| `Classes` | float | ✅ | Fire classification |
| `Region` | float | ✅ | Region identifier |

**Response:** HTML page displaying `THE FWI prediction is {result}`

---

## 📐 Input Features Reference

| Feature | Full Name | Range / Unit | Description |
|---------|-----------|-------------|-------------|
| **Temperature** | Temperature | °C | Noon temperature at time of observation |
| **RH** | Relative Humidity | 0–100 % | Moisture content in the air |
| **Ws** | Wind Speed | km/h | Sustained wind speed |
| **Rain** | Rainfall | mm | Cumulative daily rainfall |
| **FFMC** | Fine Fuel Moisture Code | 0–101 | Moisture of surface litter and fine fuels |
| **DMC** | Duff Moisture Code | 0+ | Moisture of loosely compacted organic layers |
| **ISI** | Initial Spread Index | 0+ | Expected rate of fire spread |
| **Classes** | Fire Classification | 0 or 1 | Whether fire occurred (1) or not (0) |
| **Region** | Region ID | 0 or 1 | Bejaia (0) or Sidi Bel-Abbes (1) |

---

## ☁️ Deployment

### AWS Elastic Beanstalk

The project includes an `.ebextensions/python.config` for AWS EB:

1. Install the [EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html)
2. Initialize and deploy:

   ```bash
   eb init -p python-3.8 forest-fire-app
   eb create forest-fire-env
   eb deploy
   ```

3. Open your app:

   ```bash
   eb open
   ```

### Heroku

A `Procfile` is included for Heroku deployment:

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Deploy:

   ```bash
   heroku login
   heroku create forest-fire-prediction
   git push heroku main
   heroku open
   ```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `2.0-EDA And FE Algerian Forest Fires.ipynb` | Exploratory Data Analysis — data cleaning, visualization, correlation analysis, feature engineering |
| `3.0-Model Training.ipynb` | Model Training — feature selection, train/test split, model comparison, Ridge Regression training, evaluation metrics |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgements

- **Dataset:** [UCI Machine Learning Repository — Algerian Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++)
- **Framework:** [Flask](https://flask.palletsprojects.com/)
- **ML Library:** [scikit-learn](https://scikit-learn.org/)
- **Inspiration:** End-to-end ML deployment best practices

---

<p align="center">
  Made with ❤️ for forest fire prevention and awareness
</p>
