# 00_data_loading.py

import pandas as pd
from pathlib import Path


def load_data(data_path: str = "data/HR_data_2.csv") -> pd.DataFrame:
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at: {data_path}")

    df = pd.read_csv(path)

    print("\n=== DATA LOADED SUCCESSFULLY ===")
    print(f"Shape: {df.shape}")

    return df


def basic_info(df: pd.DataFrame) -> None:
    print("\n=== BASIC INFO ===")
    print(df.info())

    print("\n=== HEAD ===")
    print(df.head())


def check_missing_values(df: pd.DataFrame) -> None:
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False))


def dataset_structure_checks(df: pd.DataFrame) -> None:
    print("\n=== DATASET STRUCTURE CHECKS ===")

    # Unique individuals
    print(f"\nNumber of unique Individuals: {df['Individual'].nunique()}")

    # Phase distribution
    print("\nPhase distribution:")
    print(df['Phase'].value_counts().sort_index())

    # Round distribution
    print("\nRound distribution:")
    print(df['Round'].value_counts().sort_index())

    # Cohort distribution (if exists)
    if 'Cohort' in df.columns:
        print("\nCohort distribution:")
        print(df['Cohort'].value_counts())

    # Team distribution
    if 'Team_ID' in df.columns:
        print("\nNumber of unique Teams:", df['Team_ID'].nunique())


def sanity_checks(df: pd.DataFrame) -> None:
    print("\n=== SANITY CHECKS ===")

    # Expected phases
    expected_phases = {'phase1', 'phase2', 'phase3'}
    found_phases = set(df['Phase'].unique())

    print(f"Phases found: {found_phases}")
    if not expected_phases.issubset(found_phases):
        print("⚠️ WARNING: Missing expected phases!")

    # Check rows per individual (should be ~12: 4 rounds x 3 phases)
    counts = df.groupby('Individual').size()
    print("\nRows per Individual (should be ~12):")
    print(counts.describe())

    # Show individuals with unusual counts
    abnormal = counts[counts != 12]
    if len(abnormal) > 0:
        print("\n⚠️ Individuals with != 12 observations:")
        print(abnormal)


def preview_key_columns(df: pd.DataFrame) -> None:
    print("\n=== PREVIEW KEY COLUMNS ===")

    cols = [
        'Individual', 'Round', 'Phase',
        'HR_TD_Mean', 'HR_TD_std',
        'EDA_TD_P_Mean', 'EDA_TD_P_std',
        'TEMP_TD_Mean'
    ]

    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].head())


def main():
    df = load_data()

    basic_info(df)
    check_missing_values(df)
    dataset_structure_checks(df)
    sanity_checks(df)
    preview_key_columns(df)


if __name__ == "__main__":
    main()