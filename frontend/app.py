import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="IntelliWatt Dashboard", layout="centered")

st.title("⚡ IntelliWatt Dashboard")
st.subheader("AI-Based Energy Analytics System")

BACKEND_URL = "http://127.0.0.1:8000"
WINDOW_SIZE = 599  # 🔥 Must match backend & training

# ==================================================
# NILM SECTION
# ==================================================
st.markdown(f"""
### 🔌 Appliance Energy Disaggregation (NILM)

1. Select an appliance  
2. Enter **exactly {WINDOW_SIZE} aggregated mains power values** (comma-separated)

📌 Sample rate: 6 seconds  
📌 599 readings ≈ 1 hour of aggregate data  

🔮 The model predicts the appliance power at the **center of that 1-hour window**
(approximately 30 minutes into the input sequence).
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
                    f"(you entered {len(data)})."
                )
            else:
                payload = {
                    "appliance": appliance.lower().replace(" ", "_"),
                    "data": data
                }

                with st.spinner("🔄 Predicting appliance power..."):
                    response = requests.post(
                        f"{BACKEND_URL}/nilm/predict",
                        json=payload,
                        timeout=20
                    )

                if response.status_code == 200:
                    result = response.json()

                    st.success("### ✅ Prediction Result")

                    st.write(f"🔌 **Appliance:** {result['appliance'].upper()}")
                    st.write(f"⚡ **Predicted Appliance Power "
                             f"(at center ≈ 30 min mark):** "
                             f"{result['predicted_power']:.2f} W"
                )

                    if result["state"] == "ON":
                        st.markdown("🟢 **State:** ON")
                    else:
                        st.markdown("🔴 **State:** OFF")

                    confidence = result.get("confidence", 0)
                    confidence_pct = confidence * 100
                    st.progress(int(confidence_pct))
                    st.write(f"📊 **Confidence:** {confidence_pct:.1f}%")

                    # ==================================================
                    # TIME-SERIES VISUALIZATION
                    # ==================================================
                    st.markdown("### 📉 Aggregate Power Window")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(data, label="Aggregate Power", color="blue")

                    center_idx = WINDOW_SIZE // 2
                    ax.axvline(
                        x=center_idx,
                        color="red",
                        linestyle="--",
                        label="Prediction Center"
                    )

                    ax.set_xlabel("Time Index")
                    ax.set_ylabel("Power (W)")
                    ax.set_title(f"{WINDOW_SIZE}-point Input Window (~1 hour) (Seq-to-Point)")
                    ax.legend()

                    st.pyplot(fig)

                    # ==================================================
                    # MODEL EXPLANATION
                    # ==================================================
                    st.markdown("### 🧠 Model Explanation")

                    pred_power = result["predicted_power"]
                    state = result["state"]
                    appliance_key = appliance.lower().replace(" ", "_")

                    if appliance_key == "kettle":
                        threshold = 1000
                        typical = "1800–2500 W"
                        behavior = "short high-power bursts"
                    elif appliance_key == "fridge":
                        threshold = 10
                        typical = "70–150 W"
                        behavior = "continuous low-power operation"
                    elif appliance_key == "washing_machine":
                        threshold = 50
                        typical = "400–2000 W (cyclic)"
                        behavior = "long cyclic power patterns"
                    elif appliance_key == "microwave":
                        threshold = 800
                        typical = "800–1500 W"
                        behavior = "short high-power bursts"
                    else:
                        threshold = None
                        typical = "unknown"
                        behavior = "unknown"

                    if state == "ON":
                        explanation = (
                            f"The model predicts that the **{appliance.lower()} is ON** because the "
                            f"predicted power (**{pred_power:.1f} W**) is above the activation "
                            f"threshold ({threshold} W). "
                            f"This aligns with the typical operating range of {typical} and "
                            f"matches the expected {behavior}."
                        )
                    else:
                        explanation = (
                            f"The model predicts that the **{appliance.lower()} is OFF** because the "
                            f"predicted power (**{pred_power:.1f} W**) is below the activation "
                            f"threshold ({threshold} W). "
                            f"This indicates no active usage at the center of the window."
                        )

                    st.info(explanation)

                else:
                    st.error(
                        f"Backend error ({response.status_code}): {response.text}"
                    )

        except requests.exceptions.Timeout:
            st.error("⏱ Backend took too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Is FastAPI running?")
        except ValueError:
            st.error("Invalid input format. Use numbers separated by commas.")

# ==================================================
# FORECASTING SECTION
# ==================================================
st.markdown("---")
st.markdown("""
### 📈 Energy Consumption Forecasting & Bill Estimation

Provide the last **60 mains power readings** (6 minutes of data).

🔮 The model predicts the **power consumption 6 seconds into the future**.
Based on that prediction, we estimate daily energy usage and monthly electricity cost.
""")
forecast_input = st.text_area(
    "Enter exactly 60 recent mains power values (comma-separated)",
    height=120
)

if st.button("Predict & Estimate Bill"):

    if forecast_input.strip() == "":
        st.error("Please enter power values.")
    else:
        try:
            data = [float(x.strip()) for x in forecast_input.split(",")]

            if len(data) != 60:
                st.error(f"Please enter exactly 60 values (you entered {len(data)}).")
            else:

                payload = {"data": data}

                with st.spinner("🔄 Predicting and estimating bill..."):
                    response = requests.post(
                        f"{BACKEND_URL}/forecast/predict",
                        json=payload,
                        timeout=10
                    )

                if response.status_code == 200:
                    result = response.json()

                    predicted_power = result["predicted_next_power_watts"]
                    daily_energy = result["estimated_daily_energy_kwh"]
                    monthly_bill = result["estimated_monthly_bill_rupees"]

                    st.success("### 🔮 Forecast Result")

                    st.write(f"⚡ **Predicted (Next 6 Seconds):** {predicted_power:.2f} W")
                    st.write(f"📊 **Estimated Daily Energy:** {daily_energy:.2f} kWh")
                    st.write(f"💰 **Estimated Monthly Bill:** ₹{monthly_bill:.2f}")

                    # ----------------------------------------
                    # Visualization
                    # ----------------------------------------
                    st.markdown("### 📉 Recent Power Window")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(data, label="Recent Power", color="blue")
                    ax.set_xlabel("Time Index")
                    ax.set_ylabel("Power (W)")
                    ax.set_title("Last 60 Data Points")
                    ax.legend()

                    st.pyplot(fig)

                    # ----------------------------------------
                    # Explanation
                    # ----------------------------------------
                    st.markdown("### 🧠 Projection Explanation")

                    explanation = (
                        f"The model predicts that your next power consumption "
                        f"will be approximately **{predicted_power:.1f} watts**. "
                        f"Assuming similar consumption continues, your estimated "
                        f"daily usage would be **{daily_energy:.2f} kWh**, "
                        f"leading to an approximate monthly bill of "
                        f"**₹{monthly_bill:.2f}** (at ₹6 per unit)."
                    )

                    st.info(explanation)

                else:
                    st.error(f"Backend error: {response.text}")

        except ValueError:
            st.error("Invalid input format. Use numbers separated by commas.")

# ==================================================
# ANOMALY DETECTION SECTION
# ==================================================
st.markdown("---")
st.markdown("""
### 🚨 Intelligent Anomaly Detection (Hybrid AI + Safety)

Provide the last **60 mains power readings** (6 minutes of data).

🧠 The system checks:
- AI-based abnormal pattern detection
- Safety limit violation (overload detection)

It classifies the behavior as:
- 🟢 Normal
- 🟡 Mild anomaly
- 🔴 Severe anomaly
""")

anomaly_input = st.text_area(
    "Enter 60 mains power values (comma-separated)",
    height=120
)

if st.button("Detect Anomaly"):

    if anomaly_input.strip() == "":
        st.error("Please enter power values.")
    else:
        try:
            data = [float(x.strip()) for x in anomaly_input.split(",")]

            if len(data) != 60:
                st.error(f"Please enter exactly 60 values (you entered {len(data)}).")
            else:

                payload = {"data": data}

                with st.spinner("🔍 Analyzing power behavior..."):
                    response = requests.post(
                        f"{BACKEND_URL}/anomaly/detect",
                        json=payload,
                        timeout=10
                    )

                if response.status_code == 200:
                    result = response.json()

                    error = result["reconstruction_error"]
                    threshold = result["threshold"]
                    max_power = result["max_power_observed"]
                    safe_limit = result["safe_limit"]
                    severity = result["severity"]

                    # ----------------------------------------
                    # Severity Display
                    # ----------------------------------------
                    if severity == "normal":
                        st.success("🟢 Normal Usage Detected")
                    elif severity == "mild":
                        st.warning("🟡 Mild Anomaly Detected")
                    else:
                        st.error("🔴 Severe Anomaly Detected")

                    # ----------------------------------------
                    # Metrics Display
                    # ----------------------------------------
                    st.write(f"📊 Reconstruction Error: {error:.4f}")
                    st.write(f"📏 Threshold: {threshold:.4f}")
                    st.write(f"⚡ Max Power Observed: {max_power:.2f} W")
                    st.write(f"🔒 Safety Limit: {safe_limit:.2f} W")

                    # ----------------------------------------
                    # Visualization
                    # ----------------------------------------
                    st.markdown("### 📉 Power Pattern (Last 6 Minutes)")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(data, label="Mains Power", color="blue")

                    # Highlight spike if exceeds safety
                    if max_power > safe_limit:
                        ax.axhline(y=safe_limit, color='red', linestyle='--', label="Safety Limit")

                    ax.set_xlabel("Time Index")
                    ax.set_ylabel("Power (W)")
                    ax.set_title("60-Point Power Window")
                    ax.legend()

                    st.pyplot(fig)

                    # ----------------------------------------
                    # Explanation
                    # ----------------------------------------
                    st.markdown("### 🧠 Explanation")

                    if severity == "normal":
                        explanation = (
                            "The recent power usage follows the learned normal household pattern. "
                            "No abnormal or unsafe behavior detected."
                        )
                    elif severity == "mild":
                        explanation = (
                            "The power pattern shows slight deviation from normal behavior. "
                            "This may indicate unusual appliance usage but not a critical issue."
                        )
                    else:
                        explanation = (
                            "The system detected significant abnormal behavior or a power spike "
                            "exceeding safe operating limits. This may indicate overload, "
                            "faulty appliance, or high-energy device usage."
                        )

                    st.info(explanation)

                else:
                    st.error(f"Backend error: {response.text}")

        except ValueError:
            st.error("Invalid input format. Use numbers separated by commas.")