from sklearn import metrics
def binary_metrics(y_true, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    return {
        "acc":  metrics.accuracy_score(y_true, preds),
        "precision": metrics.precision_score(y_true, preds),
        "recall": metrics.recall_score(y_true, preds),
        "f1": metrics.f1_score(y_true, preds),
        "auroc": metrics.roc_auc_score(y_true, probs),
    }
