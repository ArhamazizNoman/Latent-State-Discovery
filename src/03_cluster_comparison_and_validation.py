import pandas as pd
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import chi2_contingency

# ==============================
# CONFIG
# ==============================
DATA_TYPE = "relative"

FILES = {
    "kmeans": f"results/kmeans_{DATA_TYPE}_k2.csv",
    "gmm": f"results/gmm_{DATA_TYPE}_k2.csv",
    "hier": f"results/hierarchical_{DATA_TYPE}_k2.csv"
}

MERGE_KEYS = ["original_ID", "Phase", "Round"]

FEATURES = [
    "HR_TD_Mean",
    "HR_TD_std",
    "EDA_TD_P_Mean",
    "EDA_TD_P_std",
    "TEMP_TD_Mean",
    "EDA_TD_P_Peaks",
    "EDA_TD_P_Slope_mean"
]

EMOTIONS = [
    "Frustrated", "upset", "hostile", "alert", "ashamed",
    "inspired", "nervous", "attentive", "afraid", "active", "determined"
]

# ==============================
# LOAD DATA
# ==============================
print("\n=== LOADING DATA ===")

kmeans = pd.read_csv(FILES["kmeans"]).rename(columns={"cluster": "cluster_kmeans"})
gmm = pd.read_csv(FILES["gmm"]).rename(columns={"cluster": "cluster_gmm"})
hier = pd.read_csv(FILES["hier"]).rename(columns={"cluster": "cluster_hier"})

print("OK Data loaded")

# ==============================
# MERGE
# ==============================
print("\n=== MERGING DATA ===")

df = kmeans.merge(gmm, on=MERGE_KEYS)
df = df.merge(hier, on=MERGE_KEYS)

print(f"OK Merged shape: {df.shape}")

# ==============================
# CLUSTER DISTRIBUTIONS
# ==============================
print("\n=== CLUSTER DISTRIBUTIONS ===")

for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ==============================
# AGREEMENT
# ==============================
print("\n=== AGREEMENT ===")

pairs = [
    ("cluster_kmeans", "cluster_gmm"),
    ("cluster_kmeans", "cluster_hier"),
    ("cluster_gmm", "cluster_hier")
]

for a, b in pairs:
    ari = adjusted_rand_score(df[a], df[b])
    nmi = normalized_mutual_info_score(df[a], df[b])
    print(f"\n{a} vs {b}: ARI={ari:.3f}, NMI={nmi:.3f}")

# ==============================
# FEATURE PROFILING
# ==============================
print("\n=== FEATURE PROFILING ===")

for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    print(f"\n--- {col} ---")
    print(df.groupby(col)[FEATURES].mean().round(3))

# ==============================
# EMOTION PROFILING
# ==============================
print("\n=== EMOTION PROFILING ===")

for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    print(f"\n--- {col} ---")
    print(df.groupby(col)[EMOTIONS].mean().round(3))

# ==============================
# 🔥 PHASE ANALYSIS (NEW - IMPORTANT)
# ==============================
print("\n=== PHASE vs CLUSTERS ===")

for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    print(f"\n--- {col} ---")
    
    ctab = pd.crosstab(df[col], df["Phase"], normalize="columns")
    print("\nNormalized by Phase:")
    print(ctab.round(3))

# ==============================
# 🔥 AUTO INTERPRETATION PHASE
# ==============================
print("\n=== AUTO INTERPRETATION (PHASE ALIGNMENT) ===")

def phase_alignment(cluster_col):
    ctab = pd.crosstab(df[cluster_col], df["Phase"], normalize="columns")
    
    print(f"\n{cluster_col}:")
    
    for phase in ctab.columns:
        dominant_cluster = ctab[phase].idxmax()
        perc = ctab[phase].max()
        print(f"Phase {phase} -> cluster {dominant_cluster} ({perc:.2f})")

for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    phase_alignment(col)

# ==============================
# CHI-SQUARE TEST: CLUSTER vs PHASE
# ==============================
print("\n=== CHI-SQUARE TEST: CLUSTER vs PHASE ===")

for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    contingency = pd.crosstab(df[col], df["Phase"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\n{col}:")
    print(f"  Chi2 = {chi2:.3f}, p = {p:.4f}, dof = {dof}")
    if p < 0.05:
        print("  Result: cluster assignment is NOT independent of phase (p < 0.05)")
    else:
        print("  Result: no significant association between cluster and phase (p >= 0.05)")

# ==============================
# STRONG DIFFERENCES
# ==============================
print("\n=== STRONG DIFFERENCES ===")

def find_differences(cluster_col, variables, threshold=0.2):
    means = df.groupby(cluster_col)[variables].mean()
    
    if means.shape[0] != 2:
        return
    
    diff = (means.iloc[0] - means.iloc[1]).abs()
    important = diff[diff > threshold].sort_values(ascending=False)
    
    print(f"\n{cluster_col}:")
    print(important.round(3))

for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    find_differences(col, FEATURES + EMOTIONS)

# ==============================
# CONSENSUS
# ==============================
print("\n=== CONSENSUS ===")

df["consensus"] = (
    (df["cluster_kmeans"] == df["cluster_gmm"]) &
    (df["cluster_kmeans"] == df["cluster_hier"])
)

print(df["consensus"].value_counts())
print(f"Agreement ratio: {df['consensus'].mean():.3f}")

# ==============================
# SAVE
# ==============================
OUTPUT_FILE = f"results/cluster_comparison_{DATA_TYPE}.csv"
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nOK Saved: {OUTPUT_FILE}")
print("\n=== DONE ===")


# ==============================
# VISUALIZATION (PCA + PLOTS)
# ==============================
print("\n=== VISUALIZATION ===")

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# create folder if not exists
os.makedirs("figures", exist_ok=True)

# ==============================
# PREP DATA FOR PCA
# ==============================
X = df[FEATURES].copy()

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100
var_total = var1 + var2

print(f"PC1: {var1:.1f}%  PC2: {var2:.1f}%  Total: {var_total:.1f}%")

pc1_label = f"PC1 ({var1:.1f}% variance)"
pc2_label = f"PC2 ({var2:.1f}% variance)"

# ==============================
# 1. PCA - COLOR BY PHASE
# ==============================
plt.figure()

for phase in df["Phase"].unique():
    subset = df[df["Phase"] == phase]
    plt.scatter(subset["PC1"], subset["PC2"], label=phase)

plt.title(f"PCA - Colored by Phase (total variance explained: {var_total:.1f}%)")
plt.xlabel(pc1_label)
plt.ylabel(pc2_label)
plt.legend()

plt.savefig("figures/pca_phase.png")
plt.close()

# ==============================
# 2. PCA - COLOR BY CLUSTERS
# ==============================
for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:

    plt.figure()

    for c in df[col].unique():
        subset = df[df[col] == c]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"{col}_{c}")

    plt.title(f"PCA - {col} (total variance explained: {var_total:.1f}%)")
    plt.xlabel(pc1_label)
    plt.ylabel(pc2_label)
    plt.legend()

    plt.savefig(f"figures/pca_{col}.png")
    plt.close()

# ==============================
# 3. BOXPLOT FEATURES BY CLUSTER
# ==============================
for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    
    plt.figure(figsize=(10, 6))
    
    df.boxplot(column=FEATURES, by=col, rot=45)
    
    plt.title(f"Features by {col}")
    plt.suptitle("")
    plt.tight_layout()
    
    plt.savefig(f"figures/boxplot_{col}.png")
    plt.close()

# ==============================
# 4. EMOTIONS BY CLUSTER (BARPLOT)
# ==============================
for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    
    means = df.groupby(col)[EMOTIONS].mean()
    
    means.T.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Emotions by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f"figures/emotions_{col}.png")
    plt.close()

# ==============================
# 5. PHASE DISTRIBUTION PER CLUSTER
# ==============================
for col in ["cluster_kmeans", "cluster_gmm", "cluster_hier"]:
    
    ctab = pd.crosstab(df[col], df["Phase"], normalize="index")
    
    ctab.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title(f"Phase distribution per {col}")
    plt.ylabel("Proportion")
    plt.tight_layout()
    
    plt.savefig(f"figures/phase_distribution_{col}.png")
    plt.close()

print("OK Plots saved in /figures")