# GenEx: A Commonsense-aware Unified Generative Framework for Explainable Cyberbullying Detection (EMNLP 2023)

This is the official repository accompanying the EMNLP 2023 full paper: [GenEx: A Commonsense-aware Unified Generative Framework for Explainable Cyberbullying Detection](https://aclanthology.org/2023.emnlp-main.1035.pdf). The repository includes the codebase and dataset for reproducibility and further research.

---

## Robustness Study Overview

As part of the reproducibility study, we evaluated the robustness of the **GenEx** model by subjecting it to various real-world perturbations in the input data. The objective was to assess the model's consistency and adaptability to challenges like linguistic variations, noise, and complex constructs such as sarcasm or indirect language.

### Perturbations Introduced

We designed synthetic perturbations to simulate real-world challenges:

- **Minor Spelling Errors:** Introduced small typos like `"bullyy"` instead of `"bully"`.
- **Code-Mixed Variations:** Replaced Hindi words with their English synonyms and vice versa.
- **Noise Injection:** Added random punctuation or characters to simulate obfuscation (e.g., `"bu11y!!"`).
- **Sarcasm or Indirect Language:** Crafted phrases with subtle context changes to test interpretive limitations.

### Observations and Insights

- **Strengths:** 
  - The model performed well on minor spelling errors and code-mixed variations, maintaining accuracy and rationale alignment.
  - Generated rationales often aligned with human annotations for simpler perturbations.

- **Limitations:**
  - Tweets containing indirect offensive terms or sarcasm were frequently misclassified as `"bully"`.
  - Noise injection led to irrelevant or hallucinated rationales, reducing agreement with human annotations.

---

## Code Modifications

To conduct the robustness study effectively, we introduced the following changes to the original codebase:

### 1. Perturbation Pipeline
- **New Preprocessing Module:** Added a module to systematically generate perturbed versions of input data for robustness testing under various scenarios.

### 2. Enhanced Evaluation Metrics
- Integrated metrics such as:
  - **Jaccard Similarity (JS):** To measure the overlap of predicted and true rationales.
  - **Hamming Distance (HD):** To evaluate token-level differences.
  - **Ratcliff-Obershelp Similarity (ROS):** To compare textual alignment of rationales.

### 3. Training Modifications
- **Noise-Augmented Training:** Added training loops for datasets augmented with noise to test robustness during learning.
- **Improved Early Stopping:** Enhanced the early stopping mechanism to prevent overfitting during extended training runs with perturbed datasets.

---

## Insights from Modifications

The modifications allowed us to:

1. **Evaluate Robustness:** Test the model's adaptability to noisy and linguistically diverse data.
2. **Improve Metrics:** Quantify the model's alignment with human annotations using additional evaluation criteria.
3. **Identify Gaps:** Highlight areas for improvement, such as handling sarcasm and improving robustness through noisy dataset training.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{DBLP:conf/emnlp/MaityJJ0B23,
  author       = {Krishanu Maity and
                  Raghav Jain and
                  Prince Jha and
                  Sriparna Saha and
                  Pushpak Bhattacharyya},
  title        = {GenEx: {A} Commonsense-aware Unified Generative Framework for Explainable
                  Cyberbullying Detection},
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2023, Singapore, December 6-10, 2023},
  pages        = {16632--16645},
  year         = {2023},
  url          = {https://aclanthology.org/2023.emnlp-main.1035},
}
