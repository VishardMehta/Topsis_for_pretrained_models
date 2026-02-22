# TOPSIS-Based Selection of Best Pre-trained Model for Text Sentence Similarity

#### Author: Hitesh Yadav | Roll No. 102317248 | Predictive Analytics Assignment-5

---

## 📖 Overview

This project implements a structured evaluation framework to identify the optimal pre-trained Sentence Transformer model for **Text Sentence Similarity** tasks using the **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** multi-criteria decision-making method.

Instead of selecting a model purely based on accuracy, this approach evaluates multiple performance indicators and ranks models based on overall suitability.

---

## 📊 Dataset

### STS Benchmark (English)

- Source: Hugging Face (`stsb_multi_mt`)
- Split Used: Test Set
- Description:
  Contains sentence pairs with human-annotated similarity scores (0–5 scale)
- Scores normalized to: 0–1
- Purpose:
  Evaluate how well model-generated cosine similarity aligns with human judgment

---

## 🤗 Models Evaluated

The following pretrained Sentence Transformer models were evaluated:

| Model             | HuggingFace ID                    | Description                  |
| ----------------- | --------------------------------- | ---------------------------- |
| MiniLM-L6         | `all-MiniLM-L6-v2`                | Lightweight, fast, efficient |
| Paraphrase-MiniLM | `paraphrase-MiniLM-L6-v2`         | Optimized for similarity     |
| MiniLM-L12        | `all-MiniLM-L12-v2`               | Deeper MiniLM variant        |
| MPNet-Base        | `all-mpnet-base-v2`               | High accuracy transformer    |
| Multi-QA MPNet    | `multi-qa-mpnet-base-dot-v1`      | Retrieval-optimized model    |
| BERT-Large-NLI    | `bert-large-nli-stsb-mean-tokens` | Large semantic model         |

---

## 📑 Evaluation Metrics Used

The following criteria were used to build the decision matrix:

| Metric               | Description                       | Impact |
| -------------------- | --------------------------------- | ------ |
| Spearman Correlation | Correlation with human similarity | +      |
| MSE                  | Mean Squared Error                | -      |
| Throughput           | Sentences processed per second    | +      |
| Model Size (MB)      | Storage size of model             | -      |

Impact Rules:

- `+` → Higher is better
- `-` → Lower is better

---

## ⚖️ TOPSIS Configuration

Weights used:
[0.4, 0.2, 0.2, 0.2]

Meaning:

- Spearman → 40%
- MSE → 20%
- Throughput → 20%
- Size → 20%

This prioritizes semantic accuracy while still considering efficiency and deployability.

---

## 📂 Generated Results

All results are automatically saved inside the `results/` folder:

### 📄 CSV Files

- `raw_metrics.csv` → Raw evaluation metrics for all models
- `final_ranking.csv` → TOPSIS score and final ranking

---

## 📈 Saved Visualizations

The following plots are generated and saved:

- `Spearman_comparison.png`
- `MSE_comparison.png`
- `Throughput_comparison.png`
- `Size_comparison.png`
- `topsis_ranking.png`
- `decision_matrix_heatmap.png`

These visualizations help analyze:

- Individual metric comparison
- Speed vs accuracy trade-offs
- Overall ranking using TOPSIS
- Performance distribution across models

---

## 🏆 Final Recommendation

The best model is the one with:

- Highest TOPSIS Score
- Rank = 1 in `final_ranking.csv`

Based on multi-criteria evaluation:

The top-ranked model provides the best balance between:

- Semantic correlation with human judgment
- Low error
- Fast inference
- Efficient model size

This ensures suitability for real-world production systems where both accuracy and efficiency matter.

---

## ▶️ How to Run

From project root:

```bash
pip install -r requirements.txt
python src/main.py
```
