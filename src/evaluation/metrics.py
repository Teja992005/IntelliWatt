import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
def power_to_state(power, threshold):
    """
    Converts power values to ON (1) / OFF (0)
    """
    return (power > threshold).astype(int)


def f1_score(y_true, y_pred):
    """
    Computes F1-score manually
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * precision * recall / (precision + recall)
