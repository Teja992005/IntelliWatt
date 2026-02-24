from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
app = FastAPI(title="IntelliWatt API")

# ==================================================
# CONFIG
# ==================================================

WINDOW_SIZE = 599

# Appliance → model & scaler mapping
APPLIANCE_MODELS = {
    "fridge": {
        "model": "src/models/nilm_fridge.h5",
        "scaler": "src/models/nilm_fridge_scaler.pkl"
    },
    "kettle": {
        "model": "src/models/nilm_kettle.h5",
        "scaler": "src/models/nilm_kettle_scaler.pkl"
    },
    "washing_machine": {  
        "model": "src/models/nilm_washing_machine.h5",
        "scaler": "src/models/nilm_washing_machine_scaler.pkl"
    },
    "microwave": {
        "model": "src/models/nilm_microwave.h5",
        "scaler": "src/models/nilm_microwave_scaler.pkl"
    }
}

MODEL_CACHE = {}
SCALER_CACHE = {}

# ==================================================
# LOAD OTHER MODELS
# ==================================================

forecast_model = load_model(
    "src/models/forecast_model.h5",
    compile=False
)

anomaly_model = load_model(
    "src/models/anomaly_model.h5",
    compile=False
)

# ==================================================
# REQUEST SCHEMAS
# ==================================================

class NILMRequest(BaseModel):
    appliance: str
    data: List[float]

class ForecastRequest(BaseModel):
    data: List[float]

class AnomalyRequest(BaseModel):
    data: List[float]

# ==================================================
# ROUTES
# ==================================================

@app.get("/")
def root():
    return {"message": "IntelliWatt backend is running"}

# ==================================================
# NILM – MULTI APPLIANCE API
# ==================================================

@app.post("/nilm/predict")
def predict_nilm(request: NILMRequest):

    appliance = request.appliance.lower()

    if appliance not in APPLIANCE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported appliance: {appliance}"
        )

    if len(request.data) != WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"NILM expects exactly {WINDOW_SIZE} input values"
        )

    # Load model lazily
    if appliance not in MODEL_CACHE:
        MODEL_CACHE[appliance] = load_model(
            APPLIANCE_MODELS[appliance]["model"],
            compile=False
        )

    if appliance not in SCALER_CACHE:
        SCALER_CACHE[appliance] = joblib.load(
            APPLIANCE_MODELS[appliance]["scaler"]
        )

    model = MODEL_CACHE[appliance]
    scaler = SCALER_CACHE[appliance]

    # Preprocess input
    x = np.array(request.data, dtype=np.float32).reshape(-1, 1)
    x = scaler.transform(x)
    x = x.reshape(1, WINDOW_SIZE, 1)

    prediction = float(model.predict(x, verbose=0)[0][0])
    prediction = max(prediction, 0.0)  # No negative power

    # Thresholds
    THRESHOLDS = {
        "fridge": 10,
        "kettle": 1000,
        "washing_machine": 50,
        "microwave": 800

    }

    threshold = THRESHOLDS[appliance]

    distance = abs(prediction - threshold)
    confidence = float(min(distance / threshold, 1.0))

    # Decide ON / OFF
    if appliance == "fridge":
        state = "ON" if prediction > 10 else "OFF"
    elif appliance == "kettle":
        state = "ON" if prediction > 1000 else "OFF"
    elif appliance == "washing_machine":
        state = "ON" if prediction > 50 else "OFF"
    elif appliance == "microwave":
        state = "ON" if prediction > 800 else "OFF"
    else:
        state = "UNKNOWN"

    return {
        "appliance": appliance,
        "predicted_power": prediction,
        "state": state,
        "confidence": confidence
    }

# ==================================================
# FORECASTING API
# ==================================================

@app.post("/forecast/predict")
def predict_forecast(request: ForecastRequest):

    if len(request.data) != 60:
        raise HTTPException(
            status_code=400,
            detail="Forecast expects exactly 60 input values"
        )

    series = np.array(request.data, dtype=np.float32)

    # Load scaler
    scaler = joblib.load("src/models/forecast_scaler.pkl")

    scaled = scaler.transform(
        series.reshape(-1, 1)
    ).reshape(1, 60, 1)

    prediction = float(
        forecast_model.predict(scaled, verbose=0)[0][0]
    )

    prediction = max(prediction, 0.0)

    # -------------------------------
    # Stable Energy Projection
    # -------------------------------

    # Convert W → kWh for 1 day
    daily_energy_kwh = (prediction / 1000) * 24

    # Assume ₹6 per unit (changeable)
    TARIFF = 6

    monthly_bill = daily_energy_kwh * 30 * TARIFF

    return {
        "predicted_next_power_watts": prediction,
        "estimated_daily_energy_kwh": daily_energy_kwh,
        "estimated_monthly_bill_rupees": monthly_bill
    }
# ==================================================
# ANOMALY DETECTION API
# ==================================================
# Load anomaly scaler
anomaly_scaler = joblib.load("src/models/anomaly_scaler.pkl")

# Load anomaly threshold
with open("metrics/anomaly_metrics.json", "r") as f:
    anomaly_config = json.load(f)

ANOMALY_THRESHOLD = anomaly_config["threshold"]
ANOMALY_WINDOW = anomaly_config["window_size"]

@app.post("/anomaly/detect")
def detect_anomaly(request: AnomalyRequest):

    if len(request.data) != ANOMALY_WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"Anomaly detection expects exactly {ANOMALY_WINDOW} values"
        )

    series = np.array(request.data, dtype=np.float32)

    # ---------------------------
    # AI Reconstruction Part
    # ---------------------------
    series_scaled = anomaly_scaler.transform(
        series.reshape(1, -1)
    ).reshape(1, ANOMALY_WINDOW, 1)

    reconstructed = anomaly_model.predict(series_scaled, verbose=0)

    error = float(np.mean(
        np.square(series_scaled - reconstructed)
    ))

    # ---------------------------
    # Safety Rule Part
    # ---------------------------
    SAFE_LIMIT = 3000  # watts

    max_power = float(np.max(series))

    # ---------------------------
    # Severity Logic (Hybrid)
    # ---------------------------
    if max_power > SAFE_LIMIT:
        severity = "severe"
        is_anomaly = True

    else:
        if error <= ANOMALY_THRESHOLD:
            severity = "normal"
            is_anomaly = False
        elif error <= 2 * ANOMALY_THRESHOLD:
            severity = "mild"
            is_anomaly = True
        else:
            severity = "severe"
            is_anomaly = True

    return {
        "reconstruction_error": error,
        "threshold": ANOMALY_THRESHOLD,
        "max_power_observed": max_power,
        "safe_limit": SAFE_LIMIT,
        "is_anomaly": is_anomaly,
        "severity": severity
    }