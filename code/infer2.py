import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, hamming_loss
from difflib import SequenceMatcher
import numpy as np
from tabulate import tabulate

# Define additional metric functions
def calculate_ratcliff_obershelp(a, b):
    """Calculate Ratcliff-Obershelp Similarity."""
    # Ensure inputs are strings
    a = str(a)
    b = str(b)
    return SequenceMatcher(None, a, b).ratio()


def evaluate_metrics(predictions, references, task):
    """Evaluate and compute all metrics."""
    metrics = {}
    metrics['Accuracy'] = accuracy_score(references, predictions)
    metrics['F1-Score'] = f1_score(references, predictions, average='macro')

    if task == "RD":  # Additional metrics for rationale detection
        metrics['Jaccard Similarity'] = jaccard_score(references, predictions, average='macro')
        metrics['Hamming Distance'] = hamming_loss(references, predictions)
        ros = [
            calculate_ratcliff_obershelp(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        metrics['Ratcliff-Obershelp Similarity'] = np.mean(ros)

    return metrics

# Add metrics computation after inference
def main():
    # Sample predictions and references for testing
    predictions = {
        "CD": ["Bully", "Non_bully", "Bully", "Bully", "Non_bully"],
        "TI": ["Positive", "Negative", "Negative", "Positive", "Positive"],
        "SA": ["Negative", "Negative", "Positive", "Negative", "Positive"],
        "RD": ["Label1", "Label2", "Label3", "Label1", "Label2"]
    }

    references = {
        "CD": ["Bully", "Bully", "Bully", "Non_bully", "Non_bully"],
        "TI": ["Positive", "Negative", "Negative", "Negative", "Positive"],
        "SA": ["Negative", "Negative", "Positive", "Positive", "Positive"],
        "RD": ["Label1", "Label2", "Label2", "Label1", "Label3"]
    }

    # Convert categorical labels into integers for RD (if needed for additional metrics)
    label_mapping = {"Label1": 0, "Label2": 1, "Label3": 2}
    rd_predictions = [label_mapping[label] for label in predictions["RD"]]
    rd_references = [label_mapping[label] for label in references["RD"]]

    # Compute metrics for all tasks
    cd_metrics = evaluate_metrics(predictions["CD"], references["CD"], task="CD")
    ti_metrics = evaluate_metrics(predictions["TI"], references["TI"], task="TI")
    sa_metrics = evaluate_metrics(predictions["SA"], references["SA"], task="SA")
    rd_metrics = evaluate_metrics(rd_predictions, rd_references, task="RD")

    # Combine and log results
    results = {
        "Task": ["CD", "TI", "SA", "RD"],
        "Accuracy": [cd_metrics["Accuracy"], ti_metrics["Accuracy"], sa_metrics["Accuracy"], rd_metrics["Accuracy"]],
        "F1-Score": [cd_metrics["F1-Score"], ti_metrics["F1-Score"], sa_metrics["F1-Score"], rd_metrics["F1-Score"]],
        "Jaccard": ["-", "-", "-", rd_metrics.get("Jaccard Similarity", "-")],
        "Hamming": ["-", "-", "-", rd_metrics.get("Hamming Distance", "-")],
        "Ratcliff-Obershelp": ["-", "-", "-", rd_metrics.get("Ratcliff-Obershelp Similarity", "-")]
    }
    print(tabulate(results, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    main()
