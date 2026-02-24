import pandas as pd
from evaluate_models import evaluate_model
from topsis import apply_topsis
from visualize import plot_metrics, plot_topsis, plot_heatmap
import os

def main():
    models = [
        "t5-small", "t5-base",
        "sshleifer/distilbart-cnn-6-6", "sshleifer/distilbart-xsum-6-6",
        "google/flan-t5-small", "google/flan-t5-base",
    ]

    results = [evaluate_model(m) for m in models]
    df = pd.DataFrame(results)

    weights = [0.30, 0.25, 0.25, 0.10, 0.10]
    impacts = ["+", "+", "+", "+", "-"]
    final = apply_topsis(df, weights, impacts)

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(base, exist_ok=True)
    df.to_csv(os.path.join(base, "raw_metrics.csv"), index=False)
    final.to_csv(os.path.join(base, "final_ranking.csv"), index=False)

    plot_metrics(df)
    plot_topsis(final)
    plot_heatmap(df)
    print("Done! Results saved in results/")

if __name__ == "__main__":
    main()