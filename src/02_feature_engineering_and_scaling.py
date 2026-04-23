"""
FEATURE ENGINEERING

This module prepares the feature matrix used for clustering.

Main responsibilities:
- Select relevant numerical features
- Separate metadata (e.g. phase, subject) from features
- Standardize features to ensure comparable scales
- Apply optional transformations if needed

Output:
- X: processed feature matrix (ready for clustering)
- meta: auxiliary variables (not used for clustering)
- scaler: fitted scaling object (for reproducibility)
"""