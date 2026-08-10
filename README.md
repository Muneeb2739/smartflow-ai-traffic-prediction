# 🚦 SmartFlow AI

### Intelligent Traffic Prediction & Recommendation System

## 📌 Overview

SmartFlow AI is a machine learning-based traffic prediction system that predicts traffic conditions using vehicle counts and time information.

The system uses a Random Forest Classifier for traffic prediction and Gemini AI to generate smart traffic management recommendations. The application is built using Streamlit.

## 🎯 Features

- 🚦 Traffic situation prediction
- 🎯 Prediction confidence
- 📊 Prediction probability visualization
- 🤖 Gemini AI traffic recommendations
- 📈 Historical traffic analysis
- 📥 Downloadable prediction report
- 📊 Dataset overview

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- Google Gemini AI
- Joblib

## 🤖 Machine Learning

### Models Evaluated

| Model | Accuracy |
|---|---:|
| Logistic Regression | 88.09% |
| Random Forest | 96.81% |

Random Forest was selected as the final model because it achieved the highest accuracy.

### Features

- Car Count
- Bike Count
- Bus Count
- Truck Count
- Hour

### Target

- Traffic Situation

## 🧠 Gemini AI

Gemini AI generates practical traffic management recommendations based on the predicted traffic situation, vehicle counts, time, and peak-hour information.

### Workflow

Traffic Data  
↓  
Random Forest Prediction  
↓  
Traffic Situation + Confidence  
↓  
Gemini AI  
↓  
Smart Traffic Recommendation

## 📱 Streamlit Dashboard

The dashboard provides:

- Dataset overview
- Traffic prediction
- Prediction confidence
- Probability chart
- AI-generated recommendation
- Historical traffic chart
- Downloadable CSV report

## 📁 Project Structure

```text
SmartFlow-AI/
│
├── app.py
├── traffic_model.pkl
├── label_encoder.pkl
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Traffic.csv
│
└── .streamlit/
    └── secrets.toml
