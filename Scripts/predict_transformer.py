# ------------------------------------------------------------
# Stock Price Forecasting using Transformer – Prediction
# ------------------------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


# =========================
# Load dataset
# =========================
with open("stock_data.pkl", "rb") as f:
    data = pickle.load(f)

print("Data shape:", data.shape)


# =========================
# Load trained model
# =========================
model = load_model("transformer_stock_model.h5")
print("✅ Model loaded successfully")


# =========================
# Parameters
# =========================
n_steps  = 60
n_future = 20
n_past   = 50
n_feat   = data.shape[1]


# =========================
# Prepare encoder input
# =========================
last_sequence = data[-n_steps:].reshape(1, n_steps, n_feat)


# =========================
# Auto-regressive prediction
# =========================
future_predictions = []

current_sequence = last_sequence.copy()

for i in range(n_future):
    pred = model.predict(current_sequence, verbose=0)
    next_step = pred[0, -1, :]
    future_predictions.append(next_step)

    # slide window
    current_sequence = np.append(
        current_sequence[:, 1:, :],
        next_step.reshape(1, 1, n_feat),
        axis=1
    )

    print(f"Step {i+1} prediction:", next_step)

future_predictions = np.array(future_predictions)


# =========================
# Prepare plot data
# =========================
past_data = data[-n_past:]
plot_future = np.vstack([past_data[-1], future_predictions])

ax1 = np.arange(len(past_data))
ax2 = np.arange(len(past_data) - 1, len(past_data) + n_future)


# =========================
# Plot (Matplotlib output)
# =========================
plt.figure(figsize=(12, 6))

plt.plot(ax1, past_data[:, 0], 'o-', alpha=0.5, label="S&P500")
plt.plot(ax1, past_data[:, 1], 'o-', alpha=0.5, label="DOW")
plt.plot(ax1, past_data[:, 2], 'o-', alpha=0.5, label="NASDAQ")

plt.plot(ax2, plot_future[:, 0], '-', linewidth=2, label="Predicted S&P500")
plt.plot(ax2, plot_future[:, 1], '-', linewidth=2, label="Predicted DOW")
plt.plot(ax2, plot_future[:, 2], '-', linewidth=2, label="Predicted NASDAQ")

plt.axvline(x=len(past_data)-1, linestyle="dashed", linewidth=2)

plt.title("Stock Price Forecasting using Transformer")
plt.xlabel("Time step")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
