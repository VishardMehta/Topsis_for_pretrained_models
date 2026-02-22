import numpy as np
import pandas as pd


def apply_topsis(data, weights, impacts):

    matrix = data.iloc[:, 1:].values.astype(float)

    # Step 1: Normalize
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

    # Step 2: Apply weights
    weighted_matrix = norm_matrix * weights

    # Step 3: Determine ideal best & worst
    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == "+":
            ideal_best.append(weighted_matrix[:, i].max())
            ideal_worst.append(weighted_matrix[:, i].min())
        else:
            ideal_best.append(weighted_matrix[:, i].min())
            ideal_worst.append(weighted_matrix[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Step 4: Calculate distances
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Closeness coefficient
    scores = dist_worst / (dist_best + dist_worst)

    data["TOPSIS Score"] = scores
    data["Rank"] = scores.argsort()[::-1].argsort() + 1

    return data.sort_values("Rank")