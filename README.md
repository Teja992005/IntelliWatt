# IntelliWatt

IntelliWatt is a smart energy analytics project built on the UK-DALE dataset. It combines appliance-level disaggregation, short-term load forecasting, anomaly detection, a FastAPI backend, and a Streamlit frontend in one Python codebase.

## What It Includes

- NILM for appliance-level power estimation
- Sequence-to-sequence CNN and BiGRU NILM research variants
- LSTM-based mains forecasting
- LSTM autoencoder-based anomaly detection
- Rule-based severe anomaly escalation using a 3000W safety limit
- FastAPI inference backend
- Streamlit dashboard frontend

## System Flow

```text
UK-DALE data
  -> preprocessing and alignment
  -> task-specific training datasets
  -> trained models and scalers
  -> FastAPI endpoints
  -> Streamlit dashboard
```

## Project Structure

```text
IntelliWatt/
|-- backend/
|   `-- app.py
|-- frontend/
|   `-- app.py
|-- data/
|   |-- processed/
|   |-- samples/
|   `-- ukdale/
|-- metrics/
|-- reports/
|-- saved_models/
|-- src/
|   |-- anomaly/
|   |-- evaluation/
|   |-- models/
|   |-- nilm_paper/
|   |-- preprocessing/
|   |-- training/
|   `-- utils/
|-- requirements.txt
`-- README.md
```

## Core Modules

### NILM

The NILM pipeline predicts appliance-level power from aggregate mains readings.

- Main appliances: `fridge`, `kettle`, `microwave`, `washing_machine`
- Input window: `599`
- Sampling rate: `6 seconds`
- Main production-style architecture: CNN-based NILM
- Research variants included: sequence-to-sequence CNN and BiGRU
- Main training scripts live in `src/training`

### Forecasting

The forecasting pipeline predicts the next mains power value from recent usage history.

- Model type: stacked LSTM
- Input window: `60`
- Output: next power step, estimated daily energy, estimated monthly bill
- Training script: `src/training/train_forecast_mains.py`

### Anomaly Detection

The anomaly pipeline uses an LSTM autoencoder to reconstruct recent mains windows and flags unusual behavior from reconstruction error.

- Model type: sequence-to-sequence LSTM autoencoder
- Input window: `60`
- Backend endpoint: `/anomaly/detect`
- Severe rule override: power above `3000W`
- Training script: `src/training/train_anomaly.py`
- Model definition: `src/models/anomaly_autoenc.py`

## Generated Artifacts

Training updates the following key files.

### Forecasting

- `src/models/forecast_model.h5`
- `src/models/forecast_scaler.pkl`
- `metrics/forecast_metrics.json`
- `reports/forecast_metrics.png`

### Anomaly Detection

- `src/models/anomaly_model.h5`
- `src/models/anomaly_scaler.pkl`
- `metrics/anomaly_metrics.json`
- `reports/anomaly_loss_curve.png`
- `reports/anomaly_error_distribution.png`

### NILM Research And Saved Models

- `saved_models/seq2seq_6sec`
- `saved_models/paper_versions`

The repo also includes NILM paper and experiment workflows for:

- sequence-to-sequence CNN
- BiGRU

## Setup

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Run training commands from the project root.

### Train Forecast Model

```bash
.\venv\Scripts\python src\training\train_forecast_mains.py
```

### Train Anomaly Model

```bash
.\venv\Scripts\python src\training\train_anomaly.py
```

### Train NILM Models

```bash
.\venv\Scripts\python src\training\train_nilm_fridge.py
.\venv\Scripts\python src\training\train_nilm_kettle.py
.\venv\Scripts\python src\training\train_nilm_microwave.py
.\venv\Scripts\python src\training\train_nilm_washing_machine.py
```

## Run The App

Start the backend:

```bash
uvicorn backend.app:app --reload
```

Start the frontend in a separate terminal:

```bash
streamlit run frontend/app.py
```

## Main Backend Endpoints

- `GET /`
- `POST /nilm/predict`
- `POST /nilm/experiments/{model_type}`
- `POST /forecast/predict`
- `POST /anomaly/detect`

## Notes

- The frontend was redesigned without changing backend contracts.
- The backend loads anomaly settings from `metrics/anomaly_metrics.json`, so retraining the anomaly model updates runtime behavior after a backend restart.
- If you retrain while the backend is already running, restart the backend so it loads the latest model and scaler files.
