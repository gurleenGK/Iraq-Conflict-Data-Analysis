# =============================================================================
# IRAQ CONFLICT ANALYSIS — MACHINE LEARNING MODEL
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[Loading dataset...]")

df = pd.read_csv("data/processed/iraq_conflict_data.csv")

# =============================================================================
# FEATURE SELECTION (X) AND TARGET (y)
# =============================================================================
print("\n[Selecting features and target...]")

# Using wounded to predict killed (simple and interpretable)
X = df[['wounded']]
y = df['killed']

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================
print("\n[Splitting dataset...]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================================
# MODEL TRAINING
# =============================================================================
print("\n[Training Linear Regression model...]")

model = LinearRegression()
model.fit(X_train, y_train)

# =============================================================================
# PREDICTION
# =============================================================================
print("\n[Making predictions...]")

y_pred = model.predict(X_test)

# =============================================================================
# PERFORMANCE EVALUATION
# =============================================================================
print("\n--- MODEL PERFORMANCE ---")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"R²  : {r2:.4f}")

# =============================================================================
# MODEL INTERPRETATION
# =============================================================================
print("\n--- MODEL INTERPRETATION ---")

print(f"Intercept : {model.intercept_:.2f}")
print(f"Coefficient (wounded → killed): {model.coef_[0]:.4f}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n[Visualizing results...]")

plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Linear Regression: Wounded vs Killed")
plt.xlabel("Wounded")
plt.ylabel("Killed")
plt.show()
