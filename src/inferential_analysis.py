# =============================================================================
# IRAQ CONFLICT ANALYSIS — INFERENTIAL ANALYSIS (T-TEST)
# =============================================================================

import pandas as pd
from scipy.stats import ttest_ind

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[Loading dataset...]")

df = pd.read_csv("data/processed/iraq_conflict_data.csv")

# =============================================================================
# SPLIT GROUPS
# =============================================================================
print("\n[Splitting data into groups...]")

suicide_attacks = df[df['suicide'] == 1]['casualties']
non_suicide_attacks = df[df['suicide'] == 0]['casualties']

print(f"Suicide attacks count     : {len(suicide_attacks)}")
print(f"Non-suicide attacks count : {len(non_suicide_attacks)}")

# =============================================================================
# PERFORM T-TEST
# =============================================================================
print("\n[Performing independent t-test...]")

t_stat, p_value = ttest_ind(
    suicide_attacks,
    non_suicide_attacks,
    equal_var=False   # safer assumption
)

print(f"T-statistic : {t_stat:.4f}")
print(f"P-value     : {p_value:.6f}")

# =============================================================================
# INTERPRETATION
# =============================================================================
alpha = 0.05

print("\n[Interpretation]")

if p_value < alpha:
    print("Result: Statistically significant difference in casualties")
else:
    print("Result: No statistically significant difference")

# =============================================================================
# MEAN COMPARISON
# =============================================================================
print("\n[Mean Comparison]")

print(f"Average casualties (Suicide)     : {suicide_attacks.mean():.2f}")
print(f"Average casualties (Non-suicide) : {non_suicide_attacks.mean():.2f}")
