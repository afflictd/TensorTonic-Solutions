import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_true - y_pred
    huber = np.mean(np.where(np.abs(error) <= delta, np.pow(error, 2) * 0.5, delta * (np.abs(error) - delta * 0.5)))
    return huber
    