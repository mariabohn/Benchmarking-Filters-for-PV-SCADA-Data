import numpy as np


def evaluate(day, outliers, inverter, filter_name):
    truth = day["is_loss"].to_numpy()
    tp = int((outliers & truth).sum())
    fp = int((outliers & ~truth).sum())
    fn = int((~outliers & truth).sum())
    tn = int((~outliers & ~truth).sum())
    has_truth = bool(truth.any())
    precision = tp / (tp + fp) if has_truth and (tp + fp) else np.nan
    recall = tp / (tp + fn) if has_truth and (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if has_truth and precision and recall else np.nan
    return {
        "inverter": inverter,
        "filter": filter_name,
        "samples": len(day),
        "removed_pct": 100 * outliers.mean(),
        "false_positive_pct": 100 * fp / max(1, fp + tn),
        "specificity_pct": 100 * tn / max(1, tn + fp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
