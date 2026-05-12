# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.layers import (
#     Input, Dense, LayerNormalization,
#     Dropout, MultiHeadAttention
# )
# from tensorflow.keras.models import Model

# # ==============================
# # LOAD DATA
# # ==============================
# with open("stock_data.pkl", "rb") as f:
#     data = pickle.load(f)

# # data shape: (time_steps, features)
# print("Data shape:", data.shape)

# # ==============================
# # CREATE SEQUENCES (Teacher Forcing Style)
# # ==============================
# def create_sequences(data, n_steps=60):
#     X, y = [], []
#     for i in range(len(data) - n_steps):
#         X.append(data[i:i+n_steps])
#         y.append(data[i+1:i+n_steps+1])
#     return np.array(X), np.array(y)

# n_steps = 60
# X, y = create_sequences(data, n_steps)

# print("X shape:", X.shape)
# print("y shape:", y.shape)

# n_feat = X.shape[2]

# # ==============================
# # TRANSFORMER BLOCK
# # ==============================
# def transformer_block(x, d_model=64, num_heads=4, d_ff=128, dropout=0.1):
#     attn_output = MultiHeadAttention(
#         num_heads=num_heads,
#         key_dim=d_model
#     )(x, x)

#     x = LayerNormalization(epsilon=1e-6)(x + attn_output)

#     ffn = Dense(d_ff, activation="relu")(x)
#     ffn = Dense(d_model)(ffn)

#     x = LayerNormalization(epsilon=1e-6)(x + ffn)
#     x = Dropout(dropout)(x)
#     return x

# # ==============================
# # MODEL ARCHITECTURE
# # ==============================
# inputs = Input(shape=(n_steps, n_feat))
# x = Dense(64)(inputs)

# # Encoder blocks
# for _ in range(2):
#     x = transformer_block(x)

# outputs = Dense(n_feat)(x)

# model = Model(inputs, outputs)
# model.compile(
#     optimizer="adam",
#     loss="mse"
# )

# model.summary()

# # ==============================
# # TRAINING
# # ==============================
# history = model.fit(
#     X, y,
#     epochs=50,
#     batch_size=32
# )

# # ==============================
# # SAVE MODEL
# # ==============================
# model.save("transformer_stock_model.h5")

# # ==============================
# # LOSS HISTORY PLOT (LIKE IMAGE)
# # ==============================
# plt.figure(figsize=(6, 4))
# plt.plot(history.history["loss"], label="Train loss")
# plt.title("Loss history")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()











import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization,
    Dropout, MultiHeadAttention, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────
# 0. Seed
# ─────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────
# 1. Hyperparameters
# ─────────────────────────────────────────
N_STEPS  = 60    # lookback window (60 trading days ~ 3 months)
N_FUTURE = 5     # predict direction for next 5 days
D_MODEL  = 64
N_HEADS  = 4
D_FF     = 128
N_BLOCKS = 2
DROPOUT  = 0.2
BATCH    = 32
EPOCHS   = 100
TRAIN_R  = 0.70
VAL_R    = 0.15

# ─────────────────────────────────────────
# 2. Load data & scaler
# ─────────────────────────────────────────
with open("stock_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("scaler_params.pkl", "rb") as f:
    sp = pickle.load(f)
    MEAN   = sp["mean"]
    STD    = sp["std"]
    LABELS = sp["labels"]

N_FEAT = data.shape[1]
print(f"Data shape : {data.shape}")
print(f"Features   : {LABELS}")
print(f"Task       : Predict UP/DOWN for next {N_FUTURE} days")

# ─────────────────────────────────────────
# 3. Sequence builder — classification labels
#    X : normalized log returns (input window)
#    y : 1 if next day return > 0 (UP), else 0 (DOWN)
# ─────────────────────────────────────────
def create_sequences(data, n_steps, n_future):
    X, y = [], []
    for i in range(len(data) - n_steps - n_future + 1):
        X.append(data[i : i + n_steps])
        # Binary label: 1=UP, 0=DOWN for each future step and each index
        future = data[i + n_steps : i + n_steps + n_future]
        y.append((future > 0).astype(np.float32))
    return np.array(X), np.array(y)

X, y = create_sequences(data, N_STEPS, N_FUTURE)
print(f"X : {X.shape}   y : {y.shape}")
print(f"Class balance (UP ratio): {y.mean():.2%}  (should be ~50%)")

# ─────────────────────────────────────────
# 4. Chronological split
# ─────────────────────────────────────────
n     = len(X)
t_end = int(n * TRAIN_R)
v_end = int(n * (TRAIN_R + VAL_R))

X_train, y_train = X[:t_end],      y[:t_end]
X_val,   y_val   = X[t_end:v_end], y[t_end:v_end]
X_test,  y_test  = X[v_end:],      y[v_end:]

print(f"Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test.shape}")

# ─────────────────────────────────────────
# 5. Positional encoding
# ─────────────────────────────────────────
def get_positional_encoding(seq_len, d_model):
    pos  = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    ang  = pos / np.power(10000, (2 * (dims // 2)) / d_model)
    ang[:, 0::2] = np.sin(ang[:, 0::2])
    ang[:, 1::2] = np.cos(ang[:, 1::2])
    return tf.cast(ang[np.newaxis], tf.float32)

# ─────────────────────────────────────────
# 6. Transformer block
# ─────────────────────────────────────────
def transformer_block(x, d_model=64, num_heads=4, d_ff=128, dropout=0.2):
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)
    ff = Dense(d_ff, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(d_model)(ff)
    x  = LayerNormalization(epsilon=1e-6)(x + ff)
    return x

# ─────────────────────────────────────────
# 7. Model — sigmoid output for binary classification
# ─────────────────────────────────────────
inputs = Input(shape=(N_STEPS, N_FEAT))
x = Dense(D_MODEL)(inputs)
x = x + get_positional_encoding(N_STEPS, D_MODEL)
for _ in range(N_BLOCKS):
    x = transformer_block(x, D_MODEL, N_HEADS, D_FF, DROPOUT)
x   = GlobalAveragePooling1D()(x)
x   = Dense(128, activation="relu")(x)
x   = Dropout(DROPOUT)(x)
out = Dense(N_FUTURE * N_FEAT, activation="sigmoid")(x)   # sigmoid → probability
out = Reshape((N_FUTURE, N_FEAT))(out)

model = Model(inputs, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
    loss="binary_crossentropy",   # classification loss
    metrics=["accuracy"]
)
model.summary()

# ─────────────────────────────────────────
# 8. Train
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, verbose=1),
    ModelCheckpoint("transformer_stock_model.h5", save_best_only=True, verbose=0)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)
print("✅ Training complete")

# ─────────────────────────────────────────
# 9. Evaluate
# ─────────────────────────────────────────
y_prob = model.predict(X_test, verbose=0)   # probabilities (0–1)
y_pred = (y_prob > 0.5).astype(int)         # threshold at 0.5 → 0 or 1
y_true = y_test.astype(int)

# Overall accuracy
overall_acc = np.mean(y_pred == y_true)
print(f"\n🎯 Overall Directional Accuracy : {overall_acc:.2%}")
print(f"   50% = random  |  >55% = useful signal\n")

# Step 1 accuracy (most important — first day forecast)
step1_acc = np.mean(y_pred[:, 0, :] == y_true[:, 0, :])
print(f"   Step 1 (Day 1) accuracy : {step1_acc:.2%}")

# Per-index accuracy
print(f"\n   Per-index (Step 1) breakdown:")
for i, lbl in enumerate(LABELS):
    da  = np.mean(y_pred[:, 0, i] == y_true[:, 0, i])
    bar = "█" * int(da * 30)
    print(f"   {lbl:12s}: {da:.2%}  {bar}")

# Per-step accuracy
print(f"\n   Per-step accuracy:")
for s in range(N_FUTURE):
    sa = np.mean(y_pred[:, s, :] == y_true[:, s, :])
    print(f"   Day {s+1} : {sa:.2%}")

# Random baseline
random_acc = 0.50
print(f"\n⚖️  Random baseline : {random_acc:.2%}")
print(f"   Model accuracy  : {overall_acc:.2%}")
print(f"   → {'✅ BEATS random' if overall_acc > random_acc else '❌ NO BETTER than random'}")

# ─────────────────────────────────────────
# 10. Save eval artifacts
# ─────────────────────────────────────────
eval_data = {
    "y_true"   : y_true,
    "y_pred"   : y_pred,
    "y_prob"   : y_prob,
    "X_test"   : X_test,
    "history"  : history.history,
    "metrics"  : {
        "overall_acc" : float(overall_acc),
        "step1_acc"   : float(step1_acc),
        "per_index"   : {
            lbl: float(np.mean(y_pred[:, 0, i] == y_true[:, 0, i]))
            for i, lbl in enumerate(LABELS)
        }
    },
    "labels"   : LABELS,
    "n_future" : N_FUTURE
}
with open("eval_data.pkl", "wb") as f:
    pickle.dump(eval_data, f)
print("✅ eval_data.pkl saved")

# ─────────────────────────────────────────
# 11. Training curves
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"],     label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_title("Loss (Binary Crossentropy)")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["accuracy"],     label="Train")
axes[1].plot(history.history["val_accuracy"], label="Val")
axes[1].axhline(0.5, color="red", linestyle="--", label="Random (50%)")
axes[1].set_title("Directional Accuracy")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
plt.show()
print("✅ training_curves.png saved")
print("\n🎯 Next step: run  app.py")
