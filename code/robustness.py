import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from scipy.spatial.distance import hamming

# Define perturbation functions
def introduce_spelling_errors(text):
    """Introduce minor spelling errors in the text."""
    text = list(text)
    if len(text) > 1:
        idx = random.randint(0, len(text) - 1)
        text[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(text)

def code_mix_variation(text, code_dict):
    """Replace some words with code-mixed versions."""
    words = text.split()
    return ' '.join([code_dict.get(word, word) for word in words])

def add_noise(text):
    """Inject noise into the text."""
    noise = ['!', '@', '#', '$', '%', '&', '*']
    idx = random.randint(0, len(text) - 1)
    return text[:idx] + random.choice(noise) + text[idx:]

def introduce_negations(text):
    """Introduce negations into the text."""
    return text.replace("is", "is not").replace("are", "are not")

def perturb_text(text, code_dict):
    """Apply a combination of perturbations to the text."""
    perturbations = [
        introduce_spelling_errors,
        lambda t: code_mix_variation(t, code_dict),
        add_noise,
        introduce_negations
    ]
    perturbation = random.choice(perturbations)
    return perturbation(text)

# Load the GenEx model (assuming a pretrained model)
def load_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("path/to/your/model")
    tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")
    return model, tokenizer

# Evaluate the model on perturbed data
def evaluate_model(model, tokenizer, texts, labels, perturb_func, code_dict=None):
    predictions = []
    perturbed_texts = []
    for text in texts:
        perturbed = perturb_func(text, code_dict)
        perturbed_texts.append(perturbed)
        inputs = tokenizer(perturbed, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(**inputs)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded)

    # Compute metrics
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    js = jaccard_score(labels, predictions, average='weighted')
    hd = np.mean([hamming(list(label), list(pred)) for label, pred in zip(labels, predictions)])

    return acc, f1, js, hd, perturbed_texts

# Load data from Excel
def load_data_from_excel(file_path):
    """
    Load texts and labels from an Excel file.
    Args:
        file_path (str): Path to the Excel file.
    Returns:
        list, list: Lists of texts and labels.
    """
    data = pd.read_excel(file_path)
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("The Excel file must contain 'text' and 'label' columns.")
    return data["text"].tolist(), data["label"].tolist()

# Main function
if __name__ == "__main__":
    # Path to the Excel file
    file_path = "BullyExplain.xlsx"  # Replace with your Excel file path

    # Load data from the Excel file
    texts, labels = load_data_from_excel(file_path)

    # Code-mixing dictionary
    code_dict = {
        "bully": "kuttiya",
        "mean": "behan",
        "unacceptable": "asweekar"
    }

    # Load model and tokenizer
    model, tokenizer = load_model()

    # Evaluate robustness
    acc, f1, js, hd, perturbed_texts = evaluate_model(
        model, tokenizer, texts, labels, perturb_text, code_dict
    )

    # Print results
    print("Perturbed Texts:", perturbed_texts[:5])  # Print only the first 5 perturbed texts for brevity
    print(f"Accuracy: {acc:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Jaccard Similarity: {js:.2f}")
    print(f"Hamming Distance: {hd:.2f}")
