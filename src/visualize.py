import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def plot_metrics(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics = ["Spearman", "MSE", "Throughput", "Size"]

    for metric in metrics:
        plt.figure()
        plt.bar(df["Model"], df[metric])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{metric} Comparison Across Models")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_comparison.png"))
        plt.close()


def plot_topsis(df):
    plt.figure()
    plt.bar(df["Model"], df["TOPSIS Score"])
    plt.xticks(rotation=45, ha="right")
    plt.title("TOPSIS Score Ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "topsis_ranking.png"))
    plt.close()


def plot_heatmap(df):
    numeric_df = df.set_index("Model")[["Spearman", "MSE", "Throughput", "Size"]]

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df, annot=True, cmap="coolwarm")
    plt.title("Decision Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "decision_matrix_heatmap.png"))
    plt.close()