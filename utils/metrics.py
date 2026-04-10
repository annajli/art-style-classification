import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def topk_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    _, top_k_preds = outputs.topk(k, dim=1)
    correct = top_k_preds.eq(labels.unsqueeze(1).expand_as(top_k_preds))
    return correct.any(dim=1).float().mean().item()


def print_classification_report(all_labels, all_preds, class_names):
    print(classification_report(all_labels, all_preds, target_names=class_names))


def plot_confusion_matrix(all_labels, all_preds, class_names, figsize=(14, 12)):
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix")

    plt.tight_layout()
    return fig
