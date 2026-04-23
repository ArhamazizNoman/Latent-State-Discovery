import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, fcluster

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

K_RANGE = range(2, 8)
LINKAGE_METHOD = "ward"  # puoi provare anche "complete", "average"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# LOAD
# =========================

def load_dataset(path):
    return pd.read_csv(path)


# =========================
# FIND BEST K
# =========================

def find_best_k(X):
    print("\n=== SEARCHING BEST K (Hierarchical) ===")

    best_k = None
    best_score = -1
    scores = []

    # costruisci dendrogramma (linkage)
    Z = linkage(X, method=LINKAGE_METHOD)

    for k in K_RANGE:
        labels = fcluster(Z, k, criterion="maxclust")

        score = silhouette_score(X, labels)
        scores.append((k, score))

        print(f"K={k} -> Silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n✔ BEST K: {best_k} (score={best_score:.4f})")

    return best_k, scores, Z


# =========================
# FINAL MODEL
# =========================

def run_hierarchical(df, dataset_name):
    print(f"\n==============================")
    print(f"DATASET: {dataset_name}")
    print(f"==============================")

    print("\nFeatures used:")
    print(CLUSTER_FEATURES)

    X = df[CLUSTER_FEATURES].values

    # ⚠️ anche se i dati sono scalati, lo rifacciamo per sicurezza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, scores, Z = find_best_k(X_scaled)

    # final clustering
    labels = fcluster(Z, best_k, criterion="maxclust")

    df_result = df.copy()
    df_result["cluster"] = labels

    # distribuzione cluster
    print("\nCluster distribution:")
    print(pd.Series(labels).value_counts())

    # save
    output_path = RESULTS_DIR / f"hierarchical_{dataset_name}_k{best_k}.csv"
    df_result.to_csv(output_path, index=False)

    print(f"\n✔ Saved: {output_path}")

    return df_result, best_k


# =========================
# MAIN
# =========================

def main():
    for name, path in DATA_PATHS.items():
        df = load_dataset(path)
        run_hierarchical(df, name)

    print("\n=== HIERARCHICAL DONE ===")


if __name__ == "__main__":
    main()