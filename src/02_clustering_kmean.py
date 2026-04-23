# 02_kmeans.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

# =========================
# CONFIG
# =========================

DATA_PATHS = {
    "global": "data/HR_data_2_clean_global.csv",
    "relative": "data/HR_data_2_clean_relative.csv"
}

CLUSTER_FEATURES = [
    "HR_TD_Mean",
    "HR_TD_std",
    "EDA_TD_P_Mean",
    "EDA_TD_P_std",
    "TEMP_TD_Mean",
    "EDA_TD_P_Peaks",
    "EDA_TD_P_Slope_mean"
]

K_RANGE = range(2, 8)  # puoi cambiare (2–6 o 2–8)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# LOAD
# =========================

def load_dataset(path):
    df = pd.read_csv(path)
    return df


# =========================
# K SELECTION
# =========================

def find_best_k(X):
    print("\n=== SEARCHING BEST K ===")

    best_k = None
    best_score = -1
    scores = []

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)
        scores.append((k, score))

        print(f"K={k} -> Silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n✔ BEST K: {best_k} (score={best_score:.4f})")

    return best_k, scores


# =========================
# FINAL MODEL
# =========================

def run_kmeans(df, dataset_name):
    print(f"\n==============================")
    print(f"DATASET: {dataset_name}")
    print(f"==============================")

    X = df[CLUSTER_FEATURES].values

    best_k, scores = find_best_k(X)

    # Fit final model
    kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    labels = kmeans.fit_predict(X)

    df_result = df.copy()
    df_result["cluster"] = labels

    # Save
    output_path = RESULTS_DIR / f"kmeans_{dataset_name}_k{best_k}.csv"
    df_result.to_csv(output_path, index=False)

    print(f"\n✔ Saved: {output_path}")

    return df_result, best_k


# =========================
# MAIN
# =========================

def main():
    for name, path in DATA_PATHS.items():
        df = load_dataset(path)
        run_kmeans(df, name)

    print("\n=== KMEANS DONE ===")


if __name__ == "__main__":
    main()