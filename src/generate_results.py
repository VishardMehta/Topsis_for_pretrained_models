import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(BASE, exist_ok=True)

data = {
    "Model": ["t5-small", "t5-base", "google/flan-t5-small", "google/flan-t5-base",
              "sshleifer/distilbart-cnn-6-6", "sshleifer/distilbart-xsum-6-6"],
    "ROUGE-1": [0.2847, 0.3215, 0.2936, 0.3358, 0.3804, 0.2912],
    "ROUGE-2": [0.0906, 0.1184, 0.0982, 0.1297, 0.1674, 0.0879],
    "ROUGE-L": [0.1928, 0.2316, 0.2043, 0.2481, 0.2854, 0.1985],
    "Throughput": [0.37, 0.14, 0.35, 0.13, 0.22, 0.24],
    "Size": [230.81, 850.88, 307.56, 990.35, 680.22, 680.22],
}
df = pd.DataFrame(data)

matrix = df.iloc[:, 1:].values.astype(float)
norm = matrix / np.sqrt((matrix ** 2).sum(axis=0))
w = [0.30, 0.25, 0.25, 0.10, 0.10]
impacts = ["+", "+", "+", "+", "-"]
wm = norm * w
ib = [wm[:, i].max() if impacts[i] == "+" else wm[:, i].min() for i in range(5)]
iw = [wm[:, i].min() if impacts[i] == "+" else wm[:, i].max() for i in range(5)]
db = np.sqrt(((wm - ib) ** 2).sum(axis=1))
dw = np.sqrt(((wm - iw) ** 2).sum(axis=1))
df["TOPSIS Score"] = np.round(dw / (db + dw), 4)
df["Rank"] = df["TOPSIS Score"].rank(ascending=False).astype(int)
final = df.sort_values("Rank")

df.to_csv(os.path.join(BASE, "raw_metrics.csv"), index=False)
final.to_csv(os.path.join(BASE, "final_ranking.csv"), index=False)

sn = [m.split("/")[-1] for m in df["Model"]]
for m in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Throughput", "Size"]:
    plt.figure(figsize=(10, 5))
    plt.bar(sn, df[m], color=sns.color_palette("viridis", len(sn)))
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{m} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, f"{m.replace('-','_')}_comparison.png"), dpi=150)
    plt.close()

sn2 = [m.split("/")[-1] for m in final["Model"]]
plt.figure(figsize=(10, 5))
plt.bar(sn2, final["TOPSIS Score"], color=sns.color_palette("coolwarm", len(sn2)))
plt.xticks(rotation=45, ha="right")
plt.title("TOPSIS Score Ranking")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "topsis_ranking.png"), dpi=150)
plt.close()

ndf = df[["ROUGE-1","ROUGE-2","ROUGE-L","Throughput","Size"]].copy()
ndf.index = sn
plt.figure(figsize=(10, 6))
sns.heatmap(ndf, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5)
plt.title("Decision Matrix Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "decision_matrix_heatmap.png"), dpi=150)
plt.close()

print("Done!")
