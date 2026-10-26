# Late Decision-Level Fusion Stacking Ensemble for Arabic Aspect-Based Sentiment Analysis

An advanced Aspect-Based Sentiment Analysis (ABSA) system designed for Arabic review texts. This project implements a **Late Decision-Level Fusion Stacking Ensemble** that combines a deep contextual Transformer model (**MARBERTv2**) with a classical linear classifier (**LinearSVC** + TF-IDF) through a **Logistic Regression** meta-learner. This is a study of a different approach from the paper "Multi-Aspect Sentiment Classification of Arabic Tourism Reviews Using BERT and Classical Machine Learning" using the same dataset.

---

## 📌 Key Architectural Features

* **Dual Base Estimators:**
  * **MARBERTv2 (Context Engine):** Multi-head Transformer encoder adapted for informal Arabic dialects, outputting temperature-scaled ($T=1.5$), calibrated binary probability vectors per aspect.
  * **LinearSVC (Keyword Engine):** Support Vector Machine trained on unigram and bigram TF-IDF sparse matrices, outputting continuous margin decision scores (`decision_function`).
* **Sparsity & Noise Handling:** Uses PyTorch gradient-nullifying masks (`-100`) to dynamically bypass missing or neutral aspects without dropping valuable review context.
* **Meta-Fusion Layer:** L1-penalized Logistic Regression meta-learner trained on out-of-fold feature matrices to dynamically weight base model predictions based on confidence level.
* **Leak-Free Data Pipeline:** Strict separation into **Sub-Train (60%)**, **Blend (20%)**, and **Test (20%)** subsets to eliminate data leakage across calibration and stacking phases.

---

## 📐 System Architecture

```
                               ┌─────────────────────────┐
                               │       Raw Dataset       │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │  Text Preprocessing &   │
                               │   Polarity Mapping      │
                               └────────────┬────────────┘
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │ (60% Sub-Train)               │ (20% Blend)                   │ (20% Test)
            ▼                               ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│     Base Training     │       │    Meta-Calibration   │       │ Out-of-Sample Test    │
│                       │       │                       │       │                       │
│  • MARBERTv2 Training │       │  • Extract MARBERTv2  │       │  • Generate Test Probs│
│  • LinearSVC Fit      │       │    Calibrated Probs   │       │    (MARBERTv2, T=1.5)  │
└───────────────────────┘       │  • Extract LinearSVC  │       │  • Generate Test SVC  │
                                │    Decision Scores    │       │    Decision Scores    │
                                └───────────┬───────────┘       └───────────┬───────────┘
                                            │                               │
                                ┌───────────▼───────────┐                   │
                                │   Stack Meta-Matrix   │                   │
                                │      (np.hstack)      │                   │
                                └───────────┬───────────┘                   │
                                            │                               │
                                ┌───────────▼───────────┐                   │
                                │   Train Meta-Learner  │                   │
                                │  (LogisticRegression) │                   │
                                └───────────┬───────────┘                   │
                                            │                               │
                                            └───────────────┬───────────────┘
                                                            │
                                                ┌───────────▼───────────┐
                                                │    Final Inference    │
                                                │  & Metric Evaluation  │
                                                └───────────────────────┘
```

---

## 🛠️ Data Processing & Pipeline Stages

### 1. Data Preprocessing & Partitioning
* **Cleaning:** Removes URLs, non-Arabic noise, character duplicates, and normalizes vocalizations (Harakat).
* **Label Mapping:** Converts text sentiment labels into explicit binary classes: `{-1: 0, 1: 1}` (where `0 = Negative`, `1 = Positive`).
* **Masking:** Flags neutral instances and missing values with index `-100` to bypass loss computation.
* **Data Splits:**
  * **Sub-Train (`60%`):** Used strictly for optimizing MARBERTv2 parameters and fitting LinearSVC boundary margins.
  * **Blend (`20%`):** Used to generate predictions from base models to train the Logistic Regression meta-learner.
  * **Test (`20%`):** Held out purely for final performance benchmarks.

### 2. Base Estimators
* **MARBERTv2 (`ubc-nlp/MARBERTv2`):**
  * Multi-head architecture with 6 linear output heads (one per aspect).
  * Hyperparameters: Learning rate = $2 \times 10^{-5}$, Batch size = 16, Epochs = 3.
  * Calibrated probabilities generated using temperature-softened Softmax ($T=1.5$).
* **LinearSVC:**
  * Uses `TfidfVectorizer(ngram_range=(1,2), max_features=10000)`.
  * Manual text oversampling applied **only** within `sub_train_df` to handle aspect imbalance.
  * Confidence measured via decision boundary distance values (`decision_function`).

### 3. Stacking Fusion & Inference
* **Feature Stacking:** Combines 2D probability outputs from MARBERTv2 with continuous decision scores from LinearSVC into a unified feature matrix ($X_{\text{meta}}$).
* **Meta-Learner:** Fits a balanced `LogisticRegression` on the blended matrix.
* **Evaluation:** Evaluates accuracy, macro precision, macro recall, and macro F1 scores across base models and the final ensemble on the unseen test set.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.8+** installed along with **PyTorch** configured for CUDA GPU acceleration.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lujxain/arabic-absa.git
   cd arabic-absa
   ```

2. **Install dependencies:**
   ```bash
   pip install torch transformers scikit-learn pandas numpy imbalanced-learn
   ```

---

## Running the Code

1. **Prepare your dataset:** Place your annotated Arabic dataset Excel file in the project root folder (e.g., `golden_data_set_processed_google_reviews-2.xlsx`). Ensure it contains your text reviews and target aspect sentiment columns.

2. **Execute the training script:**
   ```bash
   python main.py
   ```

---

## 📊 Results & Dashboard

Upon completion, the execution script outputs a comparative benchmark matrix comparing individual base models against the decision-level stacking ensemble:

<table>
  <thead>
    <tr>
      <th>Binary Aspect</th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Macro Precision</th>
      <th>Macro Recall</th>
      <th>Macro F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>PRICE</b></td>
      <td>MARBERTv2 Only</td>
      <td>0.9713</td>
      <td>0.9319</td>
      <td>0.9555</td>
      <td>0.9432</td>
    </tr>
    <tr>
      <td>LinearSVC Only</td>
      <td>0.8947</td>
      <td>0.9453</td>
      <td>0.6333</td>
      <td>0.6816</td>
    </tr>
    <tr>
      <td>Binary Stacking</td>
      <td>0.9665</td>
      <td>0.9186</td>
      <td>0.9527</td>
      <td>0.9346</td>
    </tr>
    <tr>
      <td rowspan="3"><b>CLEANLINESS</b></td>
      <td>MARBERTv2 Only</td>
      <td>0.9570</td>
      <td>0.9485</td>
      <td>0.9598</td>
      <td>0.9536</td>
    </tr>
    <tr>
      <td>LinearSVC Only</td>
      <td>0.9140</td>
      <td>0.9412</td>
      <td>0.8788</td>
      <td>0.8998</td>
    </tr>
    <tr>
      <td>Binary Stacking</td>
      <td>0.9355</td>
      <td>0.9235</td>
      <td>0.9432</td>
      <td>0.9313</td>
    </tr>
    <tr>
      <td rowspan="3"><b>FACILITIES</b></td>
      <td>MARBERTv2 Only</td>
      <td>0.9364</td>
      <td>0.9268</td>
      <td>0.9373</td>
      <td>0.9316</td>
    </tr>
    <tr>
      <td>LinearSVC Only</td>
      <td>0.8856</td>
      <td>0.8841</td>
      <td>0.8633</td>
      <td>0.8720</td>
    </tr>
    <tr>
      <td>Binary Stacking</td>
      <td>0.9407</td>
      <td>0.9320</td>
      <td>0.9406</td>
      <td>0.9360</td>
    </tr>
    <tr>
      <td rowspan="3"><b>SERVICE</b></td>
      <td>MARBERTv2 Only</td>
      <td>0.9375</td>
      <td>0.9253</td>
      <td>0.9352</td>
      <td>0.9300</td>
    </tr>
    <tr>
      <td>LinearSVC Only</td>
      <td>0.8984</td>
      <td>0.8910</td>
      <td>0.8757</td>
      <td>0.8826</td>
    </tr>
    <tr>
      <td>Binary Stacking</td>
      <td>0.9375</td>
      <td>0.9253</td>
      <td>0.9352</td>
      <td>0.9300</td>
    </tr>
    <tr>
      <td rowspan="3"><b>ENVIRONMENT</b></td>
      <td>MARBERTv2 Only</td>
      <td>0.9606</td>
      <td>0.8846</td>
      <td>0.9404</td>
      <td>0.9098</td>
    </tr>
    <tr>
      <td>LinearSVC Only</td>
      <td>0.9507</td>
      <td>0.9352</td>
      <td>0.8106</td>
      <td>0.8592</td>
    </tr>
    <tr>
      <td>Binary Stacking</td>
      <td>0.9688</td>
      <td>0.9036</td>
      <td>0.9575</td>
      <td>0.9282</td>
    </tr>
    <tr>
      <td rowspan="3"><b>OVERALL</b></td>
      <td>MARBERTv2 Only</td>
      <td>0.9579</td>
      <td>0.9375</td>
      <td>0.9595</td>
      <td>0.9476</td>
    </tr>
    <tr>
      <td>LinearSVC Only</td>
      <td>0.9307</td>
      <td>0.9179</td>
      <td>0.9027</td>
      <td>0.9099</td>
    </tr>
    <tr>
      <td>Binary Stacking</td>
      <td>0.9653</td>
      <td>0.9492</td>
      <td>0.9646</td>
      <td>0.9565</td>
    </tr>
  </tbody>
</table>

---

## Citations

Zaid, S., Alharbi, A. H., & Samra, H. (2025). Multi-Aspect Sentiment Classification of Arabic Tourism Reviews Using BERT and Classical Machine Learning. Data, 10(11), 168. https://doi.org/10.3390/data10110168
