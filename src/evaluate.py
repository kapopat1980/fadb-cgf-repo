"""
Evaluation utilities: accuracy, macro-F1, per-class metrics, confusion matrix,
and the glioma<->meningioma confusion count used throughout the manuscript.
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)

from src.config import CLASSES, CLASS_TO_IDX


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for spatial_x, freq_x, labels in test_loader:
            spatial_x, freq_x = spatial_x.to(device), freq_x.to(device)
            outputs = model(spatial_x, freq_x)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=range(len(CLASSES)))
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASSES)))

    gm_confusion = (cm[CLASS_TO_IDX["glioma"], CLASS_TO_IDX["meningioma"]]
                    + cm[CLASS_TO_IDX["meningioma"], CLASS_TO_IDX["glioma"]])

    return {
        "accuracy": acc, "macro_f1": macro_f1,
        "per_class_precision": prec.tolist(), "per_class_recall": rec.tolist(), "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(), "glioma_meningioma_confusion": int(gm_confusion),
        "predictions": all_preds.tolist(), "labels": all_labels.tolist(),
    }
