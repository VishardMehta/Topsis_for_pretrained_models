import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def plot_metrics(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Throughput", "Size"]

    # Use short model names for readable plots
    short_names = [m.split("/")[-1] for m in df["Model"]]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.bar(short_names, df[metric], color=sns.color_palette("viridis", len(short_names)))
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.title(f"{metric} Comparison Across Models", fontsize=13)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric.replace('-', '_')}_comparison.png"), dpi=150)
        plt.close()


def plot_topsis(df):
    short_names = [m.split("/")[-1] for m in df["Model"]]
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("coolwarm", len(short_names))
    plt.bar(short_names, df["TOPSIS Score"], color=colors)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.title("TOPSIS Score Ranking", fontsize=13)
    plt.ylabel("TOPSIS Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "topsis_ranking.png"), dpi=150)
    plt.close()


def plot_heatmap(df):
    short_names = [m.split("/")[-1] for m in df["Model"]]
    numeric_df = df[["ROUGE-1", "ROUGE-2", "ROUGE-L", "Throughput", "Size"]].copy()
    numeric_df.index = short_names

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5)
    plt.title("Decision Matrix Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "decision_matrix_heatmap.png"), dpi=150)
    plt.close()