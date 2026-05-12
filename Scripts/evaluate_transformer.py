import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# LOAD DATA (same as training)
# ==============================
with open("stock_data.pkl", "rb") as f:
    data = pickle.load(f)

print("Data shape:", data.shape)

# ==============================
# CREATE SEQUENCES (same logic)
# ==============================
def create_sequences(data, n_steps=60):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+1:i+n_steps+1])
    return np.array(X), np.array(y)

n_steps = 60
X, y = create_sequences(data, n_steps)

# Train / Test split (same convention)
split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

print("Test shape:", X_test.shape, y_test.shape)

# ==============================
# LOAD TRAINED MODEL
# ==============================
model = load_model("transformer_stock_model.h5", compile=False)
print("✅ Model loaded successfully")

# ==============================
# PREDICTION
# ==============================
y_pred = model.predict(X_test, verbose=0)

# ==============================
# FLATTEN FOR METRICS
# ==============================
y_test_flat = y_test.reshape(-1, y_test.shape[-1])
y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

# ==============================
# REGRESSION METRICS
# ==============================
mse = mean_squared_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n📊 Model Evaluation Metrics")
print("----------------------------")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.6f}")

# ==============================
# VISUALIZATION
# ==============================

# Use only last 300 timesteps for clarity
steps = 300
actual = y_test_flat[-steps:]
predicted = y_pred_flat[-steps:]

labels = ["S&P 500", "DOW", "NASDAQ"]

# --------- Plot 1: Actual vs Predicted (Line Plot) ----------
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(actual[:, i], label=f"Actual {labels[i]}", alpha=0.7)
    plt.plot(predicted[:, i], '--', label=f"Predicted {labels[i]}")
plt.title("Actual vs Predicted Stock Prices (Transformer)")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True)
plt.show()

# --------- Plot 2: Residual Error ----------
plt.figure(figsize=(12, 5))
residuals = actual - predicted
for i in range(3):
    plt.plot(residuals[:, i], label=f"{labels[i]} Error")
plt.axhline(0, linestyle="dashed")
plt.title("Prediction Error (Residuals)")
plt.xlabel("Time Steps")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

# --------- Plot 3: Predicted vs Actual Scatter ----------
plt.figure(figsize=(6, 6))
plt.scatter(actual[:, 0], predicted[:, 0], alpha=0.4)
plt.xlabel("Actual S&P 500")
plt.ylabel("Predicted S&P 500")
plt.title("Predicted vs Actual (S&P 500)")
plt.grid(True)
plt.show()









# ---------------------------------------------
# Transformer Stock Price Model Evaluation
# ---------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# Load data
# =========================
with open("stock_data.pkl", "rb") as f:
    data = pickle.load(f)

print("Data shape:", data.shape)

# =========================
# Create sequences
# =========================
def create_sequences(data, n_steps=60):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+1:i+n_steps+1])
    return np.array(X), np.array(y)

X, y = create_sequences(data)

# Train-test split
split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

# =========================
# Load trained model
# =========================
model = load_model("transformer_stock_model.h5", compile=False)
print("✅ Model loaded successfully")

# =========================
# Prediction
# =========================
y_pred = model.predict(X_test, verbose=0)

# Flatten for metric calculation
y_test_f = y_test.reshape(-1, y_test.shape[-1])
y_pred_f = y_pred.reshape(-1, y_pred.shape[-1])



# =========================
mse  = mean_squared_error(y_test_f, y_pred_f)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_f, y_pred_f)
r2   = r2_score(y_test_f, y_pred_f)

print("\n📊 Model Performance:")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.6f}")

# =========================
# Plot Actual vs Predicted
# =========================
plt.figure(figsize=(10,5))
plt.plot(y_test_f[:300,0], label="Actual S&P500")
plt.plot(y_pred_f[:300,0], label="Predicted S&P500")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True)
plt.show()
