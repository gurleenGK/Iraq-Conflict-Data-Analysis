# =============================================================================
# IRAQ CONFLICT ANALYSIS — PREPROCESSING (FINAL CLEAN VERSION)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RAW_DATA_PATH = "globalterrorismdb_0718dist.csv"
OUTPUT_PATH   = "iraq_conflict_data.csv"

print("=" * 60)
print("  IRAQ CONFLICT DATA — PREPROCESSING")
print("=" * 60)

# =============================================================================
# STEP 1: Load dataset
# =============================================================================
print("\n[Step 1] Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1', low_memory=False)

# =============================================================================
# INITIAL DATA INSPECTION
# =============================================================================
print("\n[Initial Data Inspection]")

print("\n--- INFO ---")
df.info()

print("\n--- DESCRIBE (NUMERIC) ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- SAMPLE DATA ---")
print(df.head())

# =============================================================================
# STEP 2: Select relevant columns
# =============================================================================
print("\n[Step 2] Selecting relevant columns...")

cols_needed = [
    'iyear', 'imonth', 'iday',
    'country_txt', 'region_txt', 'provstate', 'city',
    'latitude', 'longitude',
    'gname',
    'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt',
    'nkill', 'nwound',
    'success', 'suicide',
    'summary'
]

df = df[[c for c in cols_needed if c in df.columns]].copy()

df.columns = [
    'year', 'month', 'day',
    'country', 'region', 'state', 'city',
    'latitude', 'longitude',
    'group_name',
    'attack_type', 'target_type', 'weapon_type',
    'killed', 'wounded',
    'success', 'suicide',
    'summary'
]

# =============================================================================
# STEP 3: Fix data types
# =============================================================================
print("\n[Step 3] Fixing data types...")

num_cols = ['year', 'month', 'day', 'killed', 'wounded']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =============================================================================
# STEP 4: Filter Iraq (2012+)
# =============================================================================
print("\n[Step 4] Filtering Iraq, year >= 2012...")

df = df[(df['country'] == 'Iraq') & (df['year'] >= 2012)].copy()

# =============================================================================
# STEP 5: Handle missing values
# =============================================================================
print("\n[Step 5] Handling missing values...")

# Numeric
df['killed']  = df['killed'].fillna(0)
df['wounded'] = df['wounded'].fillna(0)

# Coordinates → drop missing
df = df.dropna(subset=['latitude', 'longitude'])

# Categorical
df['city']        = df['city'].fillna('Unknown')
df['state']       = df['state'].fillna('Unknown')
df['weapon_type'] = df['weapon_type'].fillna('Unknown')
df['attack_type'] = df['attack_type'].fillna('Unknown')
df['target_type'] = df['target_type'].fillna('Unknown')
df['summary']     = df['summary'].fillna('No summary available')
df['group_name']  = df['group_name'].fillna('Unknown')

# =============================================================================
# STEP 6: Feature engineering
# =============================================================================
print("\n[Step 6] Creating features...")

# Casualties
df['casualties'] = df['killed'] + df['wounded']

# Fix date components
m = df['month'].replace(0, 1).fillna(1).astype(int)
d = df['day'].replace(0, 1).fillna(1).astype(int)

df['date'] = pd.to_datetime(
    dict(year=df['year'], month=m, day=d),
    errors='coerce'
)

# Derived time features
df['month_name'] = df['date'].dt.month_name()
df['day_of_week'] = df['date'].dt.day_name()

# Log transform (numerically stable)
df['log_casualties'] = np.log1p(df['casualties'])

# =============================================================================
# STEP 7: Remove zero-casualty rows
# =============================================================================
print("\n[Step 7] Removing zero-casualty incidents...")
df = df[df['casualties'] > 0]

# =============================================================================
# STEP 8: Remove unknown groups
# =============================================================================
print("\n[Step 8] Removing unknown groups...")
df = df[df['group_name'] != 'Unknown']

# =============================================================================
# STEP 9: Keep top 5 groups
# =============================================================================
print("\n[Step 9] Keeping top 5 groups...")

top5 = df['group_name'].value_counts().head(5).index
df = df[df['group_name'].isin(top5)]

print("         Groups retained:")
for grp, cnt in df['group_name'].value_counts().items():
    print(f"           {grp}: {cnt:,}")

# =============================================================================
# STEP 10: Remove duplicates
# =============================================================================
print("\n[Step 10] Removing duplicates...")

before = len(df)
df = df.drop_duplicates(
    subset=['date', 'city', 'group_name', 'attack_type']
)
print(f"         Removed: {before - len(df)} duplicates")

# =============================================================================
# STEP 11: Final column order
# =============================================================================
print("\n[Step 11] Finalizing dataset...")

final_cols = [
    'date', 'year', 'month', 'month_name', 'day', 'day_of_week',
    'country', 'region', 'state', 'city',
    'latitude', 'longitude',
    'group_name', 'attack_type', 'target_type', 'weapon_type',
    'killed', 'wounded', 'casualties', 'log_casualties',
    'success', 'suicide',
    'summary'
]

df = df[final_cols].reset_index(drop=True)

# =============================================================================
# STEP 12: Save dataset
# =============================================================================
df.to_csv(OUTPUT_PATH, index=False)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE")
print("=" * 60)

print(f"  Output file      : {OUTPUT_PATH}")
print(f"  Final shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Years covered    : {sorted(df['year'].unique())}")
print(f"  Date range       : {df['date'].min().date()} → {df['date'].max().date()}")

print("\n  Casualty Summary:")
print(f"    Total killed   : {int(df['killed'].sum()):,}")
print(f"    Total wounded  : {int(df['wounded'].sum()):,}")
print(f"    Avg casualties : {df['casualties'].mean():.2f}")
print(f"    Max casualties : {int(df['casualties'].max()):,}")

print(f"\n  Missing values   : {df.isnull().sum().sum()}")
print("=" * 60)
