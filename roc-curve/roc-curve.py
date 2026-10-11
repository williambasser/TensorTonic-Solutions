import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    """
    We need to compute the the FPR, TPR at all possible thresholds.
    The possible thresholds are discretised by the unique scores. i.e if we have y_true = [0, 1, 1], y_score = [0.2, 0.8, 0.6].
    At a t=0.6, TP=2, FP=0, TN=1, FN=0. However the TPR, FPR does not change between 0.6 and 0.8, and therefore we can define the thresholds using 0, 0.2, 0.6, 0.8
    """

    # First convert our scores to numpy arrays so we can do vectorised operations
    y_true = np.asarray(y_true)
    positives = y_true == 1 
    negatives = y_true == 0

    # Total positives and negatives
    P = positives.sum()
    N = negatives.sum()
    y_score = np.asarray(y_score)

    # Arrays for storing the points in the graph? 
    fpr_array = []
    tpr_array = []

    thresholds = np.concatenate([
        [np.inf],
        np.unique(y_score)[::-1] # Reversed so it goes in a descending order
    ])
    for threshold in thresholds:
        y_pred = y_score >= threshold
        TP = (y_pred & positives).sum() # When both are 1
        FP = (y_pred & negatives).sum()
        TPR = TP / P
        FPR = FP / N

        tpr_array.append(TPR)
        fpr_array.append(FPR)
    
    return {"tpr": np.asarray(tpr_array), "fpr": np.asarray(fpr_array), "thresholds": thresholds}
