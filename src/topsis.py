import numpy as np
import pandas as pd


def apply_topsis(data, weights, impacts):
    matrix = data.iloc[:, 1:].values.astype(float)
    norm = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted = norm * weights

    ideal_best, ideal_worst = [], []
    for i in range(len(impacts)):
        if impacts[i] == "+":
            ideal_best.append(weighted[:, i].max())
            ideal_worst.append(weighted[:, i].min())
        else:
            ideal_best.append(weighted[:, i].min())
            ideal_worst.append(weighted[:, i].max())

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    scores = dist_worst / (dist_best + dist_worst)

    data["TOPSIS Score"] = scores
    data["Rank"] = scores.argsort()[::-1].argsort() + 1
    return data.sort_values("Rank")