import numpy as np
def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    n = len(y_true)
    y_true: np.ndarray = np.asarray(y_true)
    y_pred: np.ndarray = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise
    if len(y_true) == 0:
        return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    if average == "micro":
        tp: int = int(np.sum(y_true == y_pred))
        # For micro, the key thing here is that  all of the metrics are the same thing, tp / n
        precision = tp / n
        recall = precision
        accuracy = precision
        f1 = precision
    elif average == "macro":
        # Macro is 1/n 
        classes = np.unique(np.r_[y_true, y_pred])
        total = len(classes)
        precision = 0
        accuracy = 0
        recall = 0
        f1 = 0
        for i in classes:
            accuracy_i, precision_i, recall_i, f1_i = single_class_metrics(y_true, y_pred, i)
            precision += precision_i
            accuracy += accuracy_i
            recall += recall_i
            f1 += f1_i
        precision /= total
        accuracy /= total
        recall /= total
        f1 /= total
    elif average == "weighted":
        # Macro is 1/n 
        classes = np.unique(np.r_[y_true, y_pred])
        total = len(classes)
        precision = 0
        accuracy = 0
        recall = 0
        f1 = 0
        for i in classes:
            w = int(np.sum(y_true == i)) / n
            accuracy_i, precision_i, recall_i, f1_i = single_class_metrics(y_true, y_pred, i)
            precision += precision_i * w
            accuracy += accuracy_i * w
            recall += recall_i * w
            f1 += f1_i * w
    elif average == "binary":
    # Blur out all predictions and true (however this only works )
        accuracy, precision, recall, f1 = single_class_metrics(y_true, y_pred, pos_label)
        
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }

def single_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos: int):
    n = len(y_true)
    tp = int(np.sum((y_pred == y_true) & (y_true == pos)))
    fp = int(np.sum((y_pred != y_true) & (y_pred == pos))) # Predicted positive, actually negative
    fn = int(np.sum((y_pred != pos) & (y_true == pos))) # Predicted negative, actually positive
    precision = tp / (tp + fp) if (tp+fp) != 0 else 0
    recall = tp / (tp + fn) if (tp+fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = np.mean(y_true == y_pred)
    return accuracy, precision, recall, f1
