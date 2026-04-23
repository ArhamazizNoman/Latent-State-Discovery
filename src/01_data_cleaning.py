# 01_data_cleaning.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================

CLUSTER_FEATURES = [
    "HR_TD_Mean",
    "HR_TD_std",
    "EDA_TD_P_Mean",
    "EDA_TD_P_std",
    "TEMP_TD_Mean",
    "EDA_TD_P_Peaks",
    "EDA_TD_P_Slope_mean"
]

SUBJECT_COLUMN = "Individual"


# =========================
# STEP 1: LOAD DATA
# =========================

def load_data(path="data/HR_data_2.csv"):
    df = pd.read_csv(path)

    print("\n=== DATA LOADED ===")
    print(f"Shape: {df.shape}")

    return df


# =========================
# STEP 2: BASIC CLEANING
# =========================

def basic_cleaning(df):
    print("\n=== BASIC CLEANING ===")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("✔ Dropped column: Unnamed: 0")

    print(f"Shape after cleaning: {df.shape}")

    return df


# =========================
# STEP 3: VALIDATE FEATURES
# =========================

def validate_features(df):
    missing_cols = [col for col in CLUSTER_FEATURES if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    print("✔ Features OK")


# =========================
# STEP 4: HANDLE MISSING
# =========================

def handle_missing(df):
    before = len(df)

    df_clean = df.dropna(subset=CLUSTER_FEATURES)

    after = len(df_clean)

    print("\n=== MISSING HANDLING ===")
    print(f"Dropped rows: {before - after}")

    return df_clean


# =========================
# STEP 5: GLOBAL SCALING (solo 7 feature)
# =========================

def global_scaling(df):
    print("\n=== GLOBAL SCALING ===")

    df_scaled = df.copy()

    scaler = StandardScaler()
    df_scaled[CLUSTER_FEATURES] = scaler.fit_transform(df[CLUSTER_FEATURES])

    return df_scaled


# =========================
# STEP 6: SUBJECT SCALING (solo 7 feature)
# =========================

def subject_scaling(df):
    print("\n=== SUBJECT-WISE SCALING ===")

    df_scaled = df.copy()

    df_scaled[CLUSTER_FEATURES] = df.groupby(SUBJECT_COLUMN)[CLUSTER_FEATURES].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return df_scaled


# =========================
# STEP 7: SAVE FILES
# =========================

def save_datasets(df_global, df_relative):
    print("\n=== SAVING DATASETS ===")

    df_global.to_csv("data/HR_data_2_clean_global.csv", index=False)
    print("✔ Saved: HR_data_2_clean_global.csv")

    df_relative.to_csv("data/HR_data_2_clean_relative.csv", index=False)
    print("✔ Saved: HR_data_2_clean_relative.csv")


# =========================
# MAIN PIPELINE
# =========================

def main():
    df = load_data()

    df = basic_cleaning(df)

    validate_features(df)

    df_clean = handle_missing(df)

    # 🔥 NOTA: NON rimuoviamo colonne
    df_global = global_scaling(df_clean)
    df_relative = subject_scaling(df_clean)

    save_datasets(df_global, df_relative)

    print("\n=== PIPELINE COMPLETED ===")

    return df_global, df_relative


if __name__ == "__main__":
    df_global, df_relative = main()