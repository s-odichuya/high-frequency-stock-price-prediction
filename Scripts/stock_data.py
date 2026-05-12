# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle

# # ==============================
# # SYMBOLS (Stooq format)
# # ==============================
# SYMBOLS = {
#     "SP500": "^spx",
#     "DOW": "^dji",
#     "NASDAQ": "^ndx"
# }

# def fetch_data(symbol):
#     url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
#     df = pd.read_csv(url)
#     df["Date"] = pd.to_datetime(df["Date"])
#     df.set_index("Date", inplace=True)
#     return df["Close"]

# print("Downloading data...")

# price = pd.DataFrame()

# for name, sym in SYMBOLS.items():
#     price[name] = fetch_data(sym)

# price.dropna(inplace=True)

# # ==============================
# # TREND CALCULATION
# # ==============================
# x = np.arange(len(price))

# for col in price.columns:
#     price[f"{col}_trend"] = np.polyval(np.polyfit(x, price[col], 1), x)
#     price[f"{col}_detrended"] = price[col] - price[f"{col}_trend"]

# # ==============================
# # PLOT 1: ORIGINAL + TREND
# # ==============================
# plt.figure(figsize=(10, 5))
# plt.plot(price["SP500"], label="SP500")
# plt.plot(price["SP500_trend"], "--", label="SP500 Trend")
# plt.title("SP500 Prices with Trend")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.legend()
# plt.show()

# # ==============================
# # PLOT 2: DETRENDED
# # ==============================
# plt.figure(figsize=(10, 5))
# plt.plot(price["SP500_detrended"], label="SP500 Detrended")
# plt.title("SP500 with Trend Removed")
# plt.xlabel("Time")
# plt.ylabel("Residual")
# plt.legend()
# plt.show()

# # ==============================
# # NORMALIZATION
# # ==============================
# rt = price[["SP500_detrended", "DOW_detrended", "NASDAQ_detrended"]].values
# rt_norm = (rt - rt.mean(axis=0)) / rt.std(axis=0)

# # ==============================
# # PLOT 3: NORMALIZED SERIES
# # ==============================
# plt.figure(figsize=(10, 5))
# plt.plot(rt_norm[:, 0], label="SP500")
# plt.plot(rt_norm[:, 1], label="DOW")
# plt.plot(rt_norm[:, 2], label="NASDAQ")
# plt.axhline(0)
# plt.title("Normalized Stock Prices")
# plt.xlabel("Time")
# plt.ylabel("Normalized Value")
# plt.legend()
# plt.show()

# # ==============================
# # SAVE DATA
# # ==============================
# with open("stock_data.pkl", "wb") as f:
#     pickle.dump(rt_norm, f)

# print("✅ Data downloaded, processed & saved successfully")












import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. Symbols
# ─────────────────────────────────────────
SYMBOLS = {
    "NIFTY50"  : "^NSEI",
    "SENSEX"   : "^BSESN",
    "NIFTYBANK": "^NSEBANK",
    "NIFTYIT"  : "^CNXIT",
    "MIDCAP"   : "^NSEMDCP50"
}

START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"

# ─────────────────────────────────────────
# 2. Download data
# ─────────────────────────────────────────
print("📥 Downloading Indian market data from Yahoo Finance...")
print(f"   Period : {START_DATE} → {END_DATE}\n")

price = pd.DataFrame()

for name, sym in SYMBOLS.items():
    try:
        df = yf.download(sym, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)
        if len(df) == 0:
            print(f"   ❌ {name} ({sym}) — No data returned")
            continue
        price[name] = df["Close"]
        print(f"   ✅ {name:12s} ({sym:15s}) — {len(df)} rows")
    except Exception as e:
        print(f"   ❌ {name} ({sym}) — Error: {e}")

# Drop rows where ANY index has missing data
price.dropna(inplace=True)
print(f"\n   Combined shape after dropna : {price.shape}")
print(f"   Date range : {price.index[0].date()} → {price.index[-1].date()}")

if price.shape[1] < 2:
    raise ValueError("❌ Less than 2 indices downloaded. Check your internet or symbols.")

LABELS = list(price.columns)
N_FEAT = len(LABELS)
print(f"   Features   : {LABELS}")

# ─────────────────────────────────────────
# 3. Save raw prices
# ─────────────────────────────────────────
with open("raw_prices.pkl", "wb") as f:
    pickle.dump(price, f)
print("\n✅ raw_prices.pkl saved")

# ─────────────────────────────────────────
# 4. Log returns  (replaces detrending)
#    log(P_t / P_t-1) — stationary, scale-free
#    Works across ALL market regimes (bull/bear/crash)
# ─────────────────────────────────────────
log_returns = np.log(price / price.shift(1)).dropna()
price = price.iloc[1:]   # align price index with returns

rt = log_returns.values  # (n_rows-1, N_FEAT)
print(f"\n📊 Log returns shape : {rt.shape}")
print(f"   Daily return stats (mean / std):")
for i, lbl in enumerate(LABELS):
    print(f"   {lbl:12s}  mean={rt[:,i].mean():.5f}  std={rt[:,i].std():.5f}")

# ─────────────────────────────────────────
# 5. Normalize — fit ONLY on train portion (first 70%)
# ─────────────────────────────────────────
train_end = int(len(rt) * 0.70)
mean = rt[:train_end].mean(axis=0)
std  = rt[:train_end].std(axis=0)
std  = np.where(std == 0, 1e-8, std)

rt_norm = (rt - mean) / std

# Clip extreme outliers at ±5 std (handles MIDCAP 2007 spike)
rt_norm = np.clip(rt_norm, -5, 5)

print(f"\n📊 Z-score normalization (train only, clipped ±5σ):")
for i, lbl in enumerate(LABELS):
    print(f"   {lbl:12s}  mean={mean[i]:.5f}  std={std[i]:.5f}")

# ─────────────────────────────────────────
# 6. Save processed data + scaler
# ─────────────────────────────────────────
with open("stock_data.pkl", "wb") as f:
    pickle.dump(rt_norm, f)

with open("scaler_params.pkl", "wb") as f:
    pickle.dump({
        "mean"  : mean,
        "std"   : std,
        "labels": LABELS,
        "n_feat": N_FEAT
    }, f)

print(f"\n✅ stock_data.pkl    saved  — shape: {rt_norm.shape}")
print(f"✅ scaler_params.pkl saved  — mean & std for inverse transform")

# ─────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────
colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

# Plot 1: Raw closing prices
fig, axes = plt.subplots(N_FEAT, 1, figsize=(13, 3 * N_FEAT), sharex=True)
fig.suptitle("Indian Stock Indices — Raw Closing Prices", fontsize=14, fontweight="bold", y=1.01)
for i, (ax, lbl) in enumerate(zip(axes, LABELS)):
    ax.plot(price.index, price[lbl], color=colors[i], linewidth=1)
    ax.set_ylabel(lbl, fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("plot_raw_prices.png", dpi=120, bbox_inches="tight")
plt.show()

# Plot 2: Log returns
fig2, axes2 = plt.subplots(N_FEAT, 1, figsize=(13, 3 * N_FEAT), sharex=True)
fig2.suptitle("Indian Stock Indices — Daily Log Returns", fontsize=14, fontweight="bold", y=1.01)
for i, (ax, lbl) in enumerate(zip(axes2, LABELS)):
    axes2[i].plot(log_returns.index, log_returns[lbl], color=colors[i], linewidth=0.8, alpha=0.8)
    axes2[i].axhline(0, color="black", linestyle="--", linewidth=0.5)
    axes2[i].set_ylabel(lbl, fontsize=9)
    axes2[i].grid(True, alpha=0.3)
axes2[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("plot_log_returns.png", dpi=120, bbox_inches="tight")
plt.show()

# Plot 3: Normalized returns (all on one chart)
fig3, ax3 = plt.subplots(figsize=(13, 5))
for i, (lbl, col) in enumerate(zip(LABELS, colors)):
    ax3.plot(log_returns.index, rt_norm[:, i], label=lbl,
             color=col, linewidth=0.8, alpha=0.8)
ax3.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax3.set_title("Indian Stock Indices — Normalized Log Returns (Z-score)", fontsize=13)
ax3.set_xlabel("Date"); ax3.set_ylabel("Z-score")
ax3.legend(); ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_normalized.png", dpi=120)
plt.show()

print("\n✅ Plots saved:")
print("   plot_raw_prices.png")
print("   plot_log_returns.png")
print("   plot_normalized.png")
print("\n🎯 Next step: run  train_transformer.py")
