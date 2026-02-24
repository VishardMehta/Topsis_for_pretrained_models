import pandas as pd
from evaluate_models import evaluate_model
from topsis import apply_topsis
from visualize import plot_metrics, plot_topsis, plot_heatmap
import os


def main():

    models = [
        "t5-small",
        "t5-base",
        "sshleifer/distilbart-cnn-6-6",
        "sshleifer/distilbart-xsum-6-6",
        "google/flan-t5-small",
        "google/flan-t5-base",
    ]

    results = []

    # Evaluate each model
    for model in models:
        metrics = evaluate_model(model)
        results.append(metrics)

    df = pd.DataFrame(results)

    # Define TOPSIS weights
    # ROUGE-1: 30%, ROUGE-2: 25%, ROUGE-L: 25%, Throughput: 10%, Size: 10%
    weights = [0.30, 0.25, 0.25, 0.10, 0.10]

    # Define impacts (+ higher better, - lower better)
    impacts = ["+", "+", "+", "+", "-"]

    final_ranking = apply_topsis(df, weights, impacts)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df.to_csv(os.path.join(RESULTS_DIR, "raw_metrics.csv"), index=False)

    final_ranking.to_csv(os.path.join(RESULTS_DIR, "final_ranking.csv"), index=False)

    plot_metrics(df)
    plot_topsis(final_ranking)
    plot_heatmap(df)

    print("\n✅ Evaluation complete! Results saved in results/ directory.")
    print(final_ranking[["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Throughput", "Size", "TOPSIS Score", "Rank"]].to_string(index=False))


if __name__ == "__main__":
    main()