import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="IntelliWatt Dashboard", layout="centered")

st.title("⚡ IntelliWatt Dashboard")
st.subheader("AI-Based Energy Analytics System")

BACKEND_URL = "http://127.0.0.1:8000"
WINDOW_SIZE = 599

# ==================================================
# NILM SECTION
# ==================================================

st.markdown(f"""
### 🔌 Appliance Energy Disaggregation (NILM)

1. Select an appliance  
2. Enter **exactly {WINDOW_SIZE} aggregated mains power values**

📌 Sample rate: 6 seconds  
📌 599 readings ≈ 1 hour  

The model predicts the appliance power at the **center of the window**.
""")

appliance = st.selectbox(
    "Select Appliance",
    ["Fridge", "Kettle", "Washing Machine", "Microwave"]
)

power_input = st.text_area(
    f"Mains Power Values ({WINDOW_SIZE} values)",
    height=150
)

if st.button("Predict Appliance Power"):

    if power_input.strip() == "":
        st.error("Please enter power values.")
    else:
        try:

            data = [float(x.strip()) for x in power_input.split(",")]

            if len(data) != WINDOW_SIZE:
                st.error(
                    f"Please enter exactly {WINDOW_SIZE} values "
                    f"(you entered {len(data)})"
                )

            else:

                payload = {
                    "appliance": appliance.lower().replace(" ", "_"),
                    "data": data
                }

                with st.spinner("Predicting appliance power..."):

                    response = requests.post(
                        f"{BACKEND_URL}/nilm/predict",
                        json=payload
                    )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Prediction Result")

                    st.write("Appliance:", result["appliance"])
                    st.write("Predicted Power:", round(result["predicted_power"],2),"W")

                    if result["state"] == "ON":
                        st.success("State: ON")
                    else:
                        st.error("State: OFF")

                    confidence = result["confidence"] * 100
                    st.progress(int(confidence))
                    st.write("Confidence:", round(confidence,1),"%")

                    fig, ax = plt.subplots()

                    ax.plot(data)

                    center = WINDOW_SIZE // 2

                    ax.axvline(
                        x=center,
                        color="red",
                        linestyle="--"
                    )

                    ax.set_title("Input Power Window")

                    st.pyplot(fig)

                else:
                    st.error(response.text)

        except:
            st.error("Invalid input format.")

# ==================================================
# NILM RESEARCH EXPERIMENTS
# ==================================================

st.markdown("---")

st.markdown("""
## 🧪 NILM Research Experiments

Compare research models used in the NILM paper.

Models available:

• 6-sec CNN (High resolution model)  
• 1-min CNN (Paper baseline)  
• 1-min BiGRU (Paper advanced model)

These experiments run **only for fridge appliance**.
""")

experiment_model = st.selectbox(
    "Select Experiment Model",
    [
        "6sec_cnn",
        "1min_cnn",
        "1min_bigru"
    ]
)

if experiment_model == "6sec_cnn":
    EXP_WINDOW = 599
else:
    EXP_WINDOW = 510

experiment_input = st.text_area(
    f"Enter {EXP_WINDOW} aggregate power values",
    height=150
)

if st.button("Run NILM Experiment"):

    if experiment_input.strip() == "":
        st.error("Please enter power values.")

    else:
        try:

            data = [float(x.strip()) for x in experiment_input.split(",")]

            if len(data) != EXP_WINDOW:

                st.error(
                    f"Model requires {EXP_WINDOW} values "
                    f"(you entered {len(data)})"
                )

            else:

                payload = {"data": data}

                with st.spinner("Running experiment..."):

                    response = requests.post(
                        f"{BACKEND_URL}/nilm/experiments/{experiment_model}",
                        json=payload
                    )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Experiment Completed")

                    st.write("Model:", result["model_type"])
                    st.write("Prediction Shape:", result["prediction_shape"])

                    prediction = result["prediction"]

                    st.markdown("### 📉 Input Aggregate Power")

                    fig_input, ax_input = plt.subplots(figsize=(8,3))

                    ax_input.plot(data, color="blue")

                    ax_input.set_xlabel("Time Index")
                    ax_input.set_ylabel("Power (W)")
                    ax_input.set_title("Aggregate Power Window")

                    st.pyplot(fig_input)


                    st.markdown("### ⚡ Predicted Appliance Power")

                    fig_pred, ax_pred = plt.subplots(figsize=(8,3))

                    if len(prediction) > 1:
                        ax_pred.plot(prediction, color="green")
                    else:
                        ax_pred.scatter([0], prediction, color="green")

                    ax_pred.set_xlabel("Time Index")
                    ax_pred.set_ylabel("Power (W)")
                    ax_pred.set_title("Predicted Appliance Power")

                    st.pyplot(fig_pred)

                else:
                    st.error(response.text)

        except:
            st.error("Invalid input format.")

# ==================================================
# FORECASTING SECTION
# ==================================================

st.markdown("---")

st.markdown("""
### 📈 Energy Consumption Forecasting

Provide **60 mains power readings**.
""")

forecast_input = st.text_area(
    "Enter 60 power values",
    height=120
)

if st.button("Predict & Estimate Bill"):

    if forecast_input.strip() == "":
        st.error("Please enter values")

    else:
        try:

            data = [float(x.strip()) for x in forecast_input.split(",")]

            if len(data) != 60:
                st.error("Enter exactly 60 values")

            else:

                payload = {"data": data}

                with st.spinner("Predicting..."):

                    response = requests.post(
                        f"{BACKEND_URL}/forecast/predict",
                        json=payload
                    )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Forecast Result")

                    st.write(
                        "Predicted Next Power:",
                        round(result["predicted_next_power_watts"],2),
                        "W"
                    )

                    st.write(
                        "Estimated Daily Energy:",
                        round(result["estimated_daily_energy_kwh"],2),
                        "kWh"
                    )

                    st.write(
                        "Estimated Monthly Bill:",
                        "₹",
                        round(result["estimated_monthly_bill_rupees"],2)
                    )

                    fig, ax = plt.subplots()

                    ax.plot(data)

                    ax.set_title("Recent Power Window")

                    st.pyplot(fig)

                else:
                    st.error(response.text)

        except:
            st.error("Invalid format.")

# ==================================================
# ANOMALY DETECTION SECTION
# ==================================================

st.markdown("---")

st.markdown("""
### 🚨 Anomaly Detection
Provide **60 recent mains power readings**.
""")

anomaly_input = st.text_area(
    "Enter 60 values",
    height=120
)

if st.button("Detect Anomaly"):

    if anomaly_input.strip() == "":
        st.error("Please enter values")

    else:
        try:

            data = [float(x.strip()) for x in anomaly_input.split(",")]

            if len(data) != 60:
                st.error("Enter exactly 60 values")

            else:

                payload = {"data": data}

                with st.spinner("Analyzing..."):

                    response = requests.post(
                        f"{BACKEND_URL}/anomaly/detect",
                        json=payload
                    )

                if response.status_code == 200:

                    result = response.json()

                    severity = result["severity"]

                    if severity == "normal":
                        st.success("Normal usage detected")
                    elif severity == "mild":
                        st.warning("Mild anomaly detected")
                    else:
                        st.error("Severe anomaly detected")

                    st.write(
                        "Reconstruction Error:",
                        round(result["reconstruction_error"],4)
                    )

                    st.write(
                        "Max Power Observed:",
                        round(result["max_power_observed"],2),
                        "W"
                    )

                    fig, ax = plt.subplots()

                    ax.plot(data)

                    ax.set_title("Power Pattern")

                    st.pyplot(fig)

                else:
                    st.error(response.text)

        except:
            st.error("Invalid format.")