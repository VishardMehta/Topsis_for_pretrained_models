import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def short(names):
    return [m.split("/")[-1] for m in names]

def plot_metrics(df):
    os.makedirs(BASE, exist_ok=True)
    names = short(df["Model"])
    for m in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Throughput", "Size"]:
        plt.figure(figsize=(10, 5))
        plt.bar(names, df[m], color=sns.color_palette("viridis", len(names)))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{m} Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, f"{m.replace('-','_')}_comparison.png"), dpi=150)
        plt.close()

def plot_topsis(df):
    names = short(df["Model"])
    plt.figure(figsize=(10, 5))
    plt.bar(names, df["TOPSIS Score"], color=sns.color_palette("coolwarm", len(names)))
    plt.xticks(rotation=45, ha="right")
    plt.title("TOPSIS Score Ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "topsis_ranking.png"), dpi=150)
    plt.close()

def plot_heatmap(df):
    ndf = df[["ROUGE-1","ROUGE-2","ROUGE-L","Throughput","Size"]].copy()
    ndf.index = short(df["Model"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(ndf, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5)
    plt.title("Decision Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "decision_matrix_heatmap.png"), dpi=150)
    plt.close()