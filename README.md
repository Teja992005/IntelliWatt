⚡ IntelliWatt – AI-Powered Smart Energy Analytics System

A complete deep learning–based smart energy monitoring platform built on the UK-DALE dataset, integrating:

🔌 NILM (Seq-to-Point CNN)

📈 LSTM Load Forecasting

🚨 Hybrid Anomaly Detection (LSTM Autoencoder + Safety Rule)

💰 Monthly Bill Estimation

⚙ FastAPI Backend

📊 Streamlit Dashboard

🏗 System Architecture
                ┌──────────────────────┐
                │   UK-DALE Dataset    │
                │   (6-sec sampling)   │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │   Preprocessing      │
                │ - Resampling         │
                │ - Scaling            │
                │ - Windowing          │
                └──────────┬───────────┘
                           ▼
        ┌─────────────── Parallel AI Inference ───────────────┐
        ▼                       ▼                            ▼
   NILM (CNN)            Forecast (LSTM)           Anomaly (AE + Rule)
        ▼                       ▼                            ▼
 Appliance Power        Future Load (t+1)      Reconstruction Error
                                                    + 3000W Rule
        └───────────────┬────────────────────────────┬───────────────┘
                        ▼                            ▼
               Bill Estimation                Severity Engine
                        ▼
                FastAPI Backend
                        ▼
                Streamlit Dashboard
📂 Project Structure
IntelliWatt/
│
├── data/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   │   ├── nilm/
│   │   ├── forecasting/
│   │   └── anomaly/
│   ├── training/
│   └── evaluation/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── streamlit_app.py
│
├── saved_models/
│
└── requirements.txt
🔬 Core Models
🔌 1. NILM – Seq-to-Point CNN

Window size: 599

Sampling rate: 6 seconds

Center-point prediction

Individual models for:

fridge

kettle

microwave

washing_machine

Goal: Disaggregate aggregate mains power into appliance-level consumption.

📈 2. Forecasting – LSTM

Window size: 60

6-second sampling

1-step ahead prediction

Used for:

Short-term load forecasting

Monthly bill estimation

🚨 3. Hybrid Anomaly Detection
LSTM Autoencoder

Trained on normal sequences

Uses reconstruction MSE

Threshold:

threshold = mean_error + 2 * std_error

Current metrics:

{
    "threshold": 0.4723,
    "mean_error": 0.0562,
    "std_error": 0.2080
}
Safety Rule
if total_power > 3000:
    severity = "severe"
Severity Logic
Condition	Severity
error < threshold	normal
threshold < error < 2×threshold	mild
error > 2×threshold	severe
power > 3000W	severe
💰 Monthly Bill Estimation

Assumption: ₹6 per kWh

Energy (kWh) = sum(power * 6 seconds) / 3600
Bill = Energy × 6

Uses forecasted load to estimate projected monthly cost.

⚙ Backend – FastAPI

Handles:

Model loading

Scaling

Inference

Hybrid anomaly logic

Bill estimation

JSON API responses

Example Endpoints
Endpoint	Description
/predict_nilm	Appliance disaggregation
/forecast	Load prediction
/estimate_bill	Monthly cost estimate
/detect_anomaly	Hybrid anomaly detection
/metrics	Model evaluation metrics

Run backend:

uvicorn backend.app:app --reload
📊 Frontend – Streamlit Dashboard

Features:

Real-time power graph

Appliance-level breakdown

Forecast visualization

Monthly bill estimate

Anomaly alerts (color-coded severity)

Run frontend:

streamlit run frontend/streamlit_app.py
🧠 AI Techniques Used

Convolutional Neural Networks (CNN)

Long Short-Term Memory (LSTM)

LSTM Autoencoder (Unsupervised)

Hybrid Rule-Based + Deep Learning Detection

🚀 Key Features

✔ Appliance-level energy disaggregation
✔ Short-term load forecasting
✔ Hybrid anomaly detection
✔ Monthly bill estimation
✔ Modular architecture
✔ Production-ready REST API
✔ Interactive dashboard

📊 Dataset

Trained on:

UK-DALE (UK Domestic Appliance-Level Electricity)

6-second resolution

Real household appliance-level data

🎯 Future Enhancements

Real-time IoT integration

Dynamic tariff pricing

Solar prediction integration

Multi-home scalability

Docker + Cloud deployment

Edge-device inference

🏆 Project Summary

IntelliWatt is a hybrid deep-learning energy analytics platform that combines NILM-based appliance disaggregation, LSTM forecasting, autoencoder-based anomaly detection, and bill estimation into a production-ready FastAPI + Streamlit system trained on 6-second UK-DALE data.