import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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

K_RANGE = range(2, 8)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# LOAD
# =========================

def load_dataset(path):
    return pd.read_csv(path)


# =========================
# MODEL SELECTION (BIC)
# =========================

def find_best_gmm(X):
    print("\n=== SEARCHING BEST GMM (BIC) ===")

    best_k = None
    best_bic = np.inf
    best_model = None
    bic_scores = []

    for k in K_RANGE:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42
        )

        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores.append((k, bic))

        print(f"K={k} -> BIC={bic:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = gmm

    print(f"\nOK BEST K: {best_k} (BIC={best_bic:.2f})")

    return best_model, best_k, bic_scores


# =========================
# RUN GMM
# =========================

def run_gmm(df, dataset_name):
    print(f"\n==============================")
    print(f"DATASET: {dataset_name}")
    print(f"==============================")

    print("\nFeatures used:")
    print(CLUSTER_FEATURES)

    X = df[CLUSTER_FEATURES].values

    model, best_k, bic_scores = find_best_gmm(X)

    # BIC plot
    ks = [s[0] for s in bic_scores]
    bic_vals = [s[1] for s in bic_scores]

    plt.figure(figsize=(6, 4))
    plt.plot(ks, bic_vals, marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    plt.title(f"BIC — GMM ({dataset_name})")
    plt.xlabel("K")
    plt.ylabel("BIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"gmm_bic_{dataset_name}.png")
    plt.close()
    print(f"Saved: figures/gmm_bic_{dataset_name}.png")

    # Hard labels
    labels = model.predict(X)

    # Soft probabilities (VERY USEFUL)
    probs = model.predict_proba(X)

    df_result = df.copy()
    df_result["cluster"] = labels

    # Save probabilities as extra columns
    for i in range(best_k):
        df_result[f"cluster_prob_{i}"] = probs[:, i]

    # Cluster distribution
    print("\nCluster distribution:")
    print(pd.Series(labels).value_counts())

    # Save
    output_path = RESULTS_DIR / f"gmm_{dataset_name}_k{best_k}.csv"
    df_result.to_csv(output_path, index=False)

    print(f"\nOK Saved: {output_path}")

    return df_result, best_k


# =========================
# MAIN
# =========================

def main():
    for name, path in DATA_PATHS.items():
        df = load_dataset(path)
        run_gmm(df, name)

    print("\n=== GMM DONE ===")


if __name__ == "__main__":
    main()