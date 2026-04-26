# 02_kmeans.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

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

def compute_gap_statistic(X, k_range=K_RANGE, n_simulations=20, random_state=42):
    rng = np.random.default_rng(random_state)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    gaps = []
    gap_stds = []

    for k in k_range:
        # Inertia on real data
        real_inertia = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_

        # Inertia on random reference datasets
        ref_inertias = []
        for _ in range(n_simulations):
            X_rand = rng.uniform(X_min, X_max, size=X.shape)
            ref_inertia = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_rand).inertia_
            ref_inertias.append(np.log(ref_inertia))

        gap = np.mean(ref_inertias) - np.log(real_inertia)
        std = np.std(ref_inertias) * np.sqrt(1 + 1 / n_simulations)
        gaps.append((k, gap))
        gap_stds.append((k, std))

        print(f"K={k} -> Gap={gap:.4f} (std={std:.4f})")

    # Select smallest K where gap(K) >= gap(K+1) - std(K+1)
    best_k_gap = list(k_range)[-1]
    for i in range(len(gaps) - 1):
        k, g = gaps[i]
        _, g_next = gaps[i + 1]
        _, s_next = gap_stds[i + 1]
        if g >= g_next - s_next:
            best_k_gap = k
            break

    print(f"\n-- Gap statistic suggests K={best_k_gap}")
    return gaps, gap_stds, best_k_gap


def find_best_k(X):
    print("\n=== SEARCHING BEST K ===")

    best_k = None
    best_score = -1
    scores = []
    inertias = []

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)
        scores.append((k, score))
        inertias.append((k, kmeans.inertia_))

        print(f"K={k} -> Silhouette={score:.4f}, Inertia={kmeans.inertia_:.2f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n-- Silhouette suggests K={best_k} (score={best_score:.4f})")

    print("\n=== GAP STATISTIC ===")
    gaps, gap_stds, best_k_gap = compute_gap_statistic(X)

    return best_k, scores, inertias, gaps, gap_stds, best_k_gap


# =========================
# FINAL MODEL
# =========================

def run_kmeans(df, dataset_name):
    print(f"\n==============================")
    print(f"DATASET: {dataset_name}")
    print(f"==============================")

    X = df[CLUSTER_FEATURES].values

    best_k, scores, inertias, gaps, gap_stds, best_k_gap = find_best_k(X)

    ks = [s[0] for s in scores]
    sil_vals = [s[1] for s in scores]
    inertia_vals = [i[1] for i in inertias]
    gap_vals = [g[1] for g in gaps]
    gap_std_vals = [s[1] for s in gap_stds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ks, sil_vals, marker="o")
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Silhouette K={best_k}")
    axes[0].set_title(f"Silhouette Score — K-means ({dataset_name})")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].legend()

    axes[1].plot(ks, inertia_vals, marker="o")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Silhouette K={best_k}")
    axes[1].set_title(f"Elbow (Inertia) — K-means ({dataset_name})")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Inertia")
    axes[1].legend()

    axes[2].errorbar(ks, gap_vals, yerr=gap_std_vals, marker="o", capsize=4)
    axes[2].axvline(best_k_gap, color="green", linestyle="--", label=f"Gap K={best_k_gap}")
    axes[2].set_title(f"Gap Statistic — K-means ({dataset_name})")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Gap")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"kmeans_selection_{dataset_name}.png")
    plt.close()
    print(f"Saved: figures/kmeans_selection_{dataset_name}.png")

    # Fit final model
    kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    labels = kmeans.fit_predict(X)

    df_result = df.copy()
    df_result["cluster"] = labels

    # Save
    output_path = RESULTS_DIR / f"kmeans_{dataset_name}_k{best_k}.csv"
    df_result.to_csv(output_path, index=False)

    print(f"\nOK Saved: {output_path}")

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