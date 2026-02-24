import numpy as np

def create_anomaly_data(series, window_size=60, max_samples=500_000):
    """
    Creates sliding windows from mains power data
    for anomaly detection.
    Trains autoencoder ONLY on normal patterns.
    """

    X = []
    limit = min(len(series) - window_size, max_samples)

    for i in range(limit):
        X.append(series[i:i + window_size])

    X = np.array(X, dtype="float32")

    # Autoencoder expects 2D input
    X = X.reshape((X.shape[0], X.shape[1]))

    return X
