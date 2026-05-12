import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="IndiaStock AI — Transformer Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# Custom CSS — Yahoo Finance inspired
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f8f9fa;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 1.5rem; max-width: 1400px; }

.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.2rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.header-title { color: white; font-size: 1.6rem; font-weight: 700; margin: 0; }
.header-subtitle { color: #a0aec0; font-size: 0.85rem; margin: 0; }
.header-badge {
    background: rgba(72,199,142,0.2); border: 1px solid #48c78e;
    color: #48c78e; padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; font-family: 'DM Mono', monospace;
}
.metric-card {
    background: white; border-radius: 10px; padding: 1.2rem 1.4rem;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 0.5rem;
}
.metric-label {
    color: #718096; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.3rem;
}
.metric-value {
    color: #1a202c; font-size: 1.6rem; font-weight: 700;
    font-family: 'DM Mono', monospace; line-height: 1.1;
}
.metric-delta-up { color: #38a169; font-size: 0.8rem; font-weight: 600; margin-top: 0.2rem; }
.section-header {
    color: #1a202c; font-size: 1.1rem; font-weight: 700;
    border-left: 4px solid #667eea; padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}
.signal-up {
    background: #f0fff4; border: 1.5px solid #68d391; color: #276749;
    padding: 0.6rem 0.5rem; border-radius: 8px; text-align: center;
    font-weight: 700; font-size: 0.85rem; margin-bottom: 0.3rem;
}
.signal-down {
    background: #fff5f5; border: 1.5px solid #fc8181; color: #9b2c2c;
    padding: 0.6rem 0.5rem; border-radius: 8px; text-align: center;
    font-weight: 700; font-size: 0.85rem; margin-bottom: 0.3rem;
}
.acc-row { display: flex; align-items: center; margin-bottom: 0.7rem; gap: 0.8rem; }
.acc-label { width: 100px; font-size: 0.8rem; font-weight: 600; color: #2d3748; font-family: 'DM Mono',monospace; }
.acc-bar-bg { flex: 1; background: #edf2f7; border-radius: 4px; height: 10px; overflow: hidden; }
.acc-bar-fill { height: 100%; border-radius: 4px; }
.acc-pct { width: 50px; font-size: 0.82rem; font-weight: 700; color: #2d3748; font-family: 'DM Mono',monospace; text-align: right; }
.divider { height: 1px; background: #e2e8f0; margin: 1.5rem 0; }
.footer-note { color: #a0aec0; font-size: 0.72rem; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS   = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
N_STEPS  = 60
N_FUTURE = 5

# ─────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE_DIR, "stock_data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler_params.pkl"), "rb") as f:
        sp = pickle.load(f)
    with open(os.path.join(BASE_DIR, "raw_prices.pkl"), "rb") as f:
        raw = pickle.load(f)
    model = load_model(os.path.join(BASE_DIR, "transformer_stock_model.h5"), compile=False)
    with open(os.path.join(BASE_DIR, "eval_data.pkl"), "rb") as f:
        ev = pickle.load(f)
    return data, sp, raw, model, ev

data, sp, raw_prices, model, eval_data = load_artifacts()
LABELS  = sp["labels"]
N_FEAT  = len(LABELS)
metrics = eval_data["metrics"]

# ─────────────────────────────────────────
# Predict next 5 days
# ─────────────────────────────────────────
last_window = data[-N_STEPS:].reshape(1, N_STEPS, N_FEAT)
probs   = model.predict(last_window, verbose=0)[0]   # (5, 5)
signals = (probs > 0.5).astype(int)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown("""
    <div class="header-bar">
        <p class="header-title">High Frequency Stock Price Prediction Using - Transformer Model</p>
        <p class="header-subtitle">Multi-Step Indian Index Direction Prediction &nbsp;·&nbsp;
        Nifty50 · Sensex · NiftyBank · NiftyIT · Midcap</p>
    </div>""", unsafe_allow_html=True)
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="header-badge">🟢 MODEL ACTIVE</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 1: Metrics
# ─────────────────────────────────────────
st.markdown('<p class="section-header"> Model Performance Overview</p>', unsafe_allow_html=True)

overall_acc = metrics["overall_acc"]
step1_acc   = metrics["step1_acc"]
per_index   = metrics["per_index"]
best_lbl    = max(per_index, key=per_index.get)
best_acc    = per_index[best_lbl]

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val, delta in zip(
    [c1, c2, c3, c4, c5],
    ["Overall Accuracy", "Day 1 Accuracy", f"Best: {best_lbl}", "Indices", "Horizon"],
    [f"{overall_acc:.1%}", f"{step1_acc:.1%}", f"{best_acc:.1%}", "5", "5 Days"],
    [f"▲ +{(overall_acc-0.5)*100:.1f}% vs random", "▲ Step 1 forecast",
     "▲ Strongest signal", "▲ NSE + BSE", "▲ 1 Trading Week"]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta-up">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 2: 5-Day Forecast Signals
# ─────────────────────────────────────────
st.markdown('<p class="section-header"> 5-Day Direction Forecast (Next Trading Week)</p>',
            unsafe_allow_html=True)

days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
header_cols = st.columns([1.5] + [1]*5)
with header_cols[0]:
    st.markdown("**Index**")
for col, day in zip(header_cols[1:], days):
    with col:
        st.markdown(f"**{day}**")

for idx_i, lbl in enumerate(LABELS):
    row = st.columns([1.5] + [1]*5)
    with row[0]:
        col = COLORS[idx_i]
        st.markdown(f'<span style="color:{col};font-weight:700;font-family:monospace">{lbl}</span>',
                    unsafe_allow_html=True)
    for d_i, cell in enumerate(row[1:]):
        with cell:
            sig  = signals[d_i, idx_i]
            prob = probs[d_i, idx_i]
            conf = prob if sig == 1 else (1 - prob)
            if sig == 1:
                st.markdown(f'<div class="signal-up">▲ UP<br><small>{conf:.0%} conf</small></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-down">▼ DOWN<br><small>{conf:.0%} conf</small></div>',
                            unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 3: Historical Charts
# ─────────────────────────────────────────
st.markdown('<p class="section-header"> Historical Price & Returns</p>', unsafe_allow_html=True)

selected = st.multiselect("Select indices:", LABELS, default=LABELS[:3])

if selected:
    tab1, tab2 = st.tabs(["Price Chart", "Daily Returns (Candlestick-style)"])

    with tab1:
        fig, ax = plt.subplots(figsize=(13, 4))
        fig.patch.set_facecolor("white"); ax.set_facecolor("#fafafa")
        for lbl in selected:
            i = LABELS.index(lbl)
            ax.plot(raw_prices.index, raw_prices[lbl], color=COLORS[i],
                    linewidth=1.5, label=lbl, alpha=0.9)
        ax.set_title("Indian Stock Indices — Closing Prices (₹)",
                     fontsize=11, fontweight="bold", color="#1a202c")
        ax.set_xlabel("Date", fontsize=9, color="#718096")
        ax.set_ylabel("Price (₹)", fontsize=9, color="#718096")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(colors="#718096", labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        log_returns = np.log(raw_prices / raw_prices.shift(1)).dropna()
        n_show = st.slider("Days", 60, 500, 252, key="ret_sl")
        fig2, axes2 = plt.subplots(len(selected), 1,
                                    figsize=(13, 2.5*len(selected)), sharex=True)
        fig2.patch.set_facecolor("white")
        if len(selected) == 1:
            axes2 = [axes2]
        for ax2, lbl in zip(axes2, selected):
            i    = LABELS.index(lbl)
            rets = log_returns[lbl].iloc[-n_show:].values
            bar_colors = ["#38a169" if r >= 0 else "#e53e3e" for r in rets]
            ax2.bar(range(len(rets)), rets, color=bar_colors, width=0.8, alpha=0.8)
            ax2.axhline(0, color="#2d3748", linewidth=0.8, linestyle="--")
            ax2.set_ylabel(lbl, fontsize=8, color=COLORS[i])
            ax2.set_facecolor("#fafafa")
            ax2.grid(True, alpha=0.2, axis="y")
            ax2.spines[["top","right"]].set_visible(False)
            ax2.tick_params(colors="#718096", labelsize=7)
        axes2[-1].set_xlabel("Trading Days (recent)", fontsize=9, color="#718096")
        fig2.suptitle("Daily Log Returns — 🟢 UP  🔴 DOWN",
                      fontsize=10, fontweight="bold", color="#1a202c", y=1.01)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 4: Accuracy + Training History
# ─────────────────────────────────────────
st.markdown('<p class="section-header"> Model Evaluation</p>', unsafe_allow_html=True)

col_acc, col_hist = st.columns([1, 1.5])

with col_acc:
    st.markdown("**Directional Accuracy per Index — Day 1**")
    st.markdown("")
    for i, lbl in enumerate(LABELS):
        acc  = per_index[lbl]
        fill = max(0, min(100, int((acc - 0.40) / 0.30 * 100)))
        beat = "✅" if acc > 0.50 else "⚠️"
        col  = COLORS[i]
        st.markdown(f"""
        <div class="acc-row">
            <span class="acc-label">{lbl}</span>
            <div class="acc-bar-bg">
                <div class="acc-bar-fill" style="width:{fill}%;
                     background:linear-gradient(90deg,{col}88,{col});"></div>
            </div>
            <span class="acc-pct">{beat} {acc:.0%}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.8rem;background:#f7fafc;
                border-radius:8px;border:1px solid #e2e8f0;">
        <div style="font-size:0.72rem;color:#718096;font-weight:600;
                    text-transform:uppercase;margin-bottom:0.3rem;">
            vs Random Baseline (50%)
        </div>
        <div style="font-size:1rem;font-weight:700;color:#38a169;
                    font-family:'DM Mono',monospace;">
            +{(overall_acc-0.5)*100:.2f}% edge over coin flip
        </div>
    </div>""", unsafe_allow_html=True)

with col_hist:
    st.markdown("**Training Loss History**")
    history = eval_data["history"]
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    fig3.patch.set_facecolor("white"); ax3.set_facecolor("#fafafa")
    ep = range(1, len(history["loss"]) + 1)
    ax3.plot(ep, history["loss"],     color="#667eea", lw=2, label="Train Loss")
    ax3.plot(ep, history["val_loss"], color="#ed8936", lw=2, linestyle="--", label="Val Loss")
    ax3.set_title("Binary Crossentropy Loss", fontsize=10, fontweight="bold", color="#1a202c")
    ax3.set_xlabel("Epoch", fontsize=8, color="#718096")
    ax3.set_ylabel("Loss", fontsize=8, color="#718096")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2, linestyle="--")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.tick_params(colors="#718096", labelsize=8)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 5: Actual vs Predicted + Confusion Matrix
# ─────────────────────────────────────────
st.markdown('<p class="section-header"> Actual vs Predicted (Test Set)</p>',
            unsafe_allow_html=True)

y_true = eval_data["y_true"]
y_pred = eval_data["y_pred"]
y_prob = eval_data["y_prob"]

sel_idx = st.selectbox("Select index:", LABELS)
ii      = LABELS.index(sel_idx)

col_main, col_cm = st.columns([2.2, 1])

with col_main:
    n_show2   = st.slider("Test samples", 50, 400, 200, key="avp_sl")
    true_d1   = y_true[-n_show2:, 0, ii]
    pred_d1   = y_pred[-n_show2:, 0, ii]
    prob_d1   = y_prob[-n_show2:, 0, ii]
    correct   = true_d1 == pred_d1
    incorrect = ~correct

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(11, 5),
                                       gridspec_kw={"height_ratios": [3,1]})
    fig4.patch.set_facecolor("white")

    ax4a.scatter(np.where(correct)[0], true_d1[correct],
                 color="#38a169", s=14, alpha=0.7, label=f"Correct ({correct.mean():.0%})", zorder=3)
    ax4a.scatter(np.where(incorrect)[0], true_d1[incorrect],
                 color="#e53e3e", s=14, alpha=0.7, label=f"Wrong ({incorrect.mean():.0%})", zorder=3)
    ax4a.step(range(n_show2), pred_d1, color="#667eea", lw=1, alpha=0.5,
              label="Predicted", where="mid")
    ax4a.set_facecolor("#fafafa"); ax4a.set_ylim(-0.3, 1.3)
    ax4a.set_yticks([0, 1]); ax4a.set_yticklabels(["▼ DOWN", "▲ UP"])
    ax4a.set_title(f"{sel_idx} — Day 1 Direction Forecast vs Actual",
                   fontsize=10, fontweight="bold", color="#1a202c")
    ax4a.legend(fontsize=8, ncol=3); ax4a.grid(True, alpha=0.15, axis="y")
    ax4a.spines[["top","right"]].set_visible(False)
    ax4a.tick_params(colors="#718096", labelsize=8)

    ax4b.fill_between(range(n_show2), 0.5, prob_d1,
                      where=prob_d1 >= 0.5, color="#38a169", alpha=0.4, label="UP confidence")
    ax4b.fill_between(range(n_show2), prob_d1, 0.5,
                      where=prob_d1 < 0.5, color="#e53e3e", alpha=0.4, label="DOWN confidence")
    ax4b.axhline(0.5, color="#718096", lw=0.8, linestyle="--")
    ax4b.set_ylabel("Probability", fontsize=8, color="#718096")
    ax4b.set_xlabel("Test Sample", fontsize=8, color="#718096")
    ax4b.set_ylim(0, 1); ax4b.set_facecolor("#fafafa")
    ax4b.spines[["top","right"]].set_visible(False)
    ax4b.tick_params(colors="#718096", labelsize=8)
    ax4b.legend(fontsize=7, ncol=2)

    plt.tight_layout(); st.pyplot(fig4); plt.close()

with col_cm:
    cm = confusion_matrix(y_true[:, 0, ii], y_pred[:, 0, ii])
    fig5, ax5 = plt.subplots(figsize=(4, 3.8))
    fig5.patch.set_facecolor("white")
    ax5.imshow(cm, cmap="Blues", aspect="auto")
    ax5.set_xticks([0, 1]); ax5.set_yticks([0, 1])
    ax5.set_xticklabels(["▼ DOWN", "▲ UP"], fontsize=9)
    ax5.set_yticklabels(["▼ DOWN", "▲ UP"], fontsize=9)
    ax5.set_xlabel("Predicted", fontsize=9, color="#718096")
    ax5.set_ylabel("Actual",    fontsize=9, color="#718096")
    ax5.set_title(f"Confusion Matrix\n{sel_idx}", fontsize=9,
                  fontweight="bold", color="#1a202c")
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, str(cm[i,j]), ha="center", va="center",
                     fontsize=15, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "#2d3748")
    plt.tight_layout(); st.pyplot(fig5); plt.close()

    tn, fp, fn, tp = cm.ravel()
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    st.markdown(f"""
    <div style="background:#f7fafc;border-radius:8px;padding:0.8rem;
                border:1px solid #e2e8f0;margin-top:0.5rem;">
        <div style="font-size:0.72rem;color:#718096;font-weight:600;
                    text-transform:uppercase;margin-bottom:0.5rem;">
            Classification Report
        </div>
        <div style="font-size:0.82rem;color:#2d3748;line-height:2;
                    font-family:'DM Mono',monospace;">
            Precision : {prec:.2%}<br>
            Recall    : {rec:.2%}<br>
            F1 Score  : {f1:.2%}<br>
            TP / TN   : {tp} / {tn}<br>
            FP / FN   : {fp} / {fn}
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 6: Recent Trends
# ─────────────────────────────────────────
st.markdown('<p class="section-header"> Recent Market Trend (Last 100 Trading Days)</p>',
            unsafe_allow_html=True)

fig6, axes6 = plt.subplots(1, N_FEAT, figsize=(14, 3))
fig6.patch.set_facecolor("white")

for i, (ax6, lbl) in enumerate(zip(axes6, LABELS)):
    prices = raw_prices[lbl].iloc[-100:].values
    change = (prices[-1] - prices[0]) / prices[0] * 100
    col    = COLORS[i]

    ax6.plot(range(100), prices, color=col, lw=1.8)
    ax6.fill_between(range(100), prices.min()*0.995, prices,
                     color=col, alpha=0.1)
    ax6.set_title(lbl, fontsize=9, fontweight="bold", color=col)
    ax6.set_facecolor("white")
    ax6.spines[["top","right","left","bottom"]].set_visible(False)
    ax6.tick_params(colors="#a0aec0", labelsize=6)
    ax6.set_xticks([])
    sign  = "▲" if change >= 0 else "▼"
    cchg  = "#38a169" if change >= 0 else "#e53e3e"
    ax6.text(0.98, 0.05, f"{sign} {abs(change):.1f}%",
             transform=ax6.transAxes, ha="right", va="bottom",
             fontsize=9, fontweight="bold", color=cchg, fontfamily="monospace")

plt.suptitle("100-Day Price Performance", fontsize=10,
             fontweight="bold", color="#1a202c", y=1.02)
plt.tight_layout()
st.pyplot(fig6)
plt.close()

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="footer-note">
    IndiaStock AI &nbsp;·&nbsp; Transformer-based Direction Forecasting &nbsp;·&nbsp;
    Trained on 2007–2024 NSE/BSE Daily Data &nbsp;·&nbsp;
    Overall Accuracy {overall_acc:.1%} &nbsp;·&nbsp; 5 Indian Indices &nbsp;·&nbsp;
    <strong>For Academic Research Only — Not Financial Advice</strong>
</div>""", unsafe_allow_html=True)
