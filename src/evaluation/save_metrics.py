import json
import os
from datetime import datetime

def save_metrics(
    appliance_name,
    mae,
    rmse,
    f1_score=None,
    output_dir="metrics"
):
    """
    Save evaluation metrics to a JSON file.
    Supports both NILM (with F1-score) and Forecasting (without F1-score).
    """

    os.makedirs(output_dir, exist_ok=True)

    metrics_data = {
        "appliance": appliance_name,
        "mae_watts": round(float(mae), 3),
        "rmse_watts": round(float(rmse), 3),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # ✅ Add F1-score only if it exists (NILM models)
    if f1_score is not None:
        metrics_data["f1_score"] = round(float(f1_score), 3)

    file_path = os.path.join(
        output_dir,
        f"{appliance_name}_metrics.json"
    )

    with open(file_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"✅ Metrics saved to: {file_path}")