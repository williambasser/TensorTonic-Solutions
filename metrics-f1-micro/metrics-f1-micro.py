import numpy as np

def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum(y_true == y_pred))
    errors = len(y_true) - tp        # fp + fn together
    f1 = tp / (tp + errors)  # == tp / len(y_true)
    return round(float(f1), 4)
