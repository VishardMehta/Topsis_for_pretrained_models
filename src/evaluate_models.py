from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import time


def evaluate_model(model_name):

    print(f"Evaluating {model_name}...")

    model = SentenceTransformer(model_name)

    dataset = load_dataset("stsb_multi_mt", name="en", split="test")

    sentences1 = dataset["sentence1"]
    sentences2 = dataset["sentence2"]
    true_scores = np.array(dataset["similarity_score"]) / 5.0

    # Measure inference time
    start = time.time()
    embeddings1 = model.encode(sentences1, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, convert_to_numpy=True)
    end = time.time()

    throughput = len(sentences1) / (end - start)

    # Cosine similarity
    cos_scores = [
        cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
        for i in range(len(embeddings1))
    ]

    spearman_corr = spearmanr(cos_scores, true_scores).correlation
    mse = mean_squared_error(true_scores, cos_scores)

    # Approx model size (MB)
    size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)

    return {
        "Model": model_name,
        "Spearman": spearman_corr,
        "MSE": mse,
        "Throughput": throughput,
        "Size": size_mb
    }