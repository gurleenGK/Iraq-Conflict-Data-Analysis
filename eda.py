# =============================================================================
# IRAQ CONFLICT ANALYSIS — EDA
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[Loading cleaned dataset...]")
df = pd.read_csv("data/processed/iraq_conflict_data.csv")

# =============================================================================
# 1) UNIVARIATE ANALYSIS (Distribution + Skewness)
# =============================================================================
print("\n--- UNIVARIATE ANALYSIS ---")

# Histogram (Casualties)
plt.figure()
sns.histplot(df['casualties'], bins=30, kde=True)
plt.title("Distribution of Casualties")
plt.xlabel("Casualties")
plt.ylabel("Frequency")
plt.show()

# Skewness
print("\nSkewness:")
for col in ['killed', 'wounded', 'casualties']:
    sk = df[col].skew()
    print(f"{col}: {sk:.3f}")
    if sk > 1:
        print(f" -> {col} is highly positively skewed")
    elif sk < -1:
        print(f" -> {col} is highly negatively skewed")
    else:
        print(f" -> {col} is approximately symmetric")

# =============================================================================
# 2) OUTLIER ANALYSIS (Boxplot + IQR)
# =============================================================================
print("\n--- OUTLIER ANALYSIS ---")

# Boxplot
plt.figure()
sns.boxplot(x=df['casualties'])
plt.title("Boxplot of Casualties")
plt.xlabel("Casualties")
plt.show()

# IQR method
Q1 = df['casualties'].quantile(0.25)
Q3 = df['casualties'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['casualties'] < lower) | (df['casualties'] > upper)]
print(f"Number of outliers (casualties): {len(outliers)}")

# =============================================================================
# 3) CATEGORICAL ANALYSIS
# =============================================================================
print("\n--- CATEGORICAL ANALYSIS ---")

# Top 5 groups (already filtered, but still useful to show counts)
plt.figure()
df['group_name'].value_counts().plot(kind='bar')
plt.title("Top 5 Terrorist Groups")
plt.ylabel("Number of Incidents")
plt.xticks(rotation=45)
plt.show()

# Attack types
plt.figure()
df['attack_type'].value_counts().head(10).plot(kind='bar')
plt.title("Top Attack Types")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Target types
plt.figure()
df['target_type'].value_counts().head(10).plot(kind='bar')
plt.title("Top Target Types")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# =============================================================================
# 4) BASIC TIME TREND (Simple but effective)
# =============================================================================
print("\n--- TIME ANALYSIS ---")

year_counts = df['year'].value_counts().sort_index()

plt.figure()
year_counts.plot(kind='line', marker='o')
plt.title("Number of Attacks per Year")
plt.xlabel("Year")
plt.ylabel("Incidents")
plt.show()
