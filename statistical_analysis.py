# =============================================================================
# IRAQ CONFLICT ANALYSIS — STATISTICAL ANALYSIS
# =============================================================================

import pandas as pd

# =============================================================================
# LOAD CLEANED DATA
# =============================================================================
print("\n[Loading cleaned dataset...]")

df = pd.read_csv("data/processed/iraq_conflict_data.csv")

# =============================================================================
# BASIC STATISTICAL ANALYSIS
# =============================================================================
print("\n--- DATA INFO ---")
df.info()

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
print("\n--- CORRELATION MATRIX ---")

corr_matrix = df[['killed', 'wounded', 'casualties', 'success', 'suicide']].corr()
print(corr_matrix)

# =============================================================================
# COVARIANCE ANALYSIS
# =============================================================================
print("\n--- COVARIANCE MATRIX ---")

cov_matrix = df[['killed', 'wounded', 'casualties']].cov()
print(cov_matrix)

# =============================================================================
# BASIC INSIGHTS
# =============================================================================
print("\n--- QUICK INSIGHTS ---")

print("Average casualties:", df['casualties'].mean())
print("Max casualties:", df['casualties'].max())
print("Min casualties:", df['casualties'].min())
