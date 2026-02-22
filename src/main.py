import pandas as pd
from evaluate_models import evaluate_model
from topsis import apply_topsis
from visualize import plot_metrics, plot_topsis, plot_heatmap
import os


def main():

    models = [
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "bert-large-nli-stsb-mean-tokens"
    ]

    results = []

    # Evaluate each model
    for model in models:
        metrics = evaluate_model(model)
        results.append(metrics)

    df = pd.DataFrame(results)

    # Define TOPSIS weights
    weights = [0.4, 0.2, 0.2, 0.2]

    # Define impacts (+ higher better, - lower better)
    impacts = ["+", "-", "+", "-"]

    final_ranking = apply_topsis(df, weights, impacts)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df.to_csv(os.path.join(RESULTS_DIR, "raw_metrics.csv"), index=False)

    final_ranking.to_csv(os.path.join(RESULTS_DIR, "final_ranking.csv"), index=False)

    plot_metrics(df)
    plot_topsis(final_ranking)
    plot_heatmap(df)


if __name__ == "__main__":
    main()