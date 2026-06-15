import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SemiYield · Wafer Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { color-scheme: dark !important; }
</style>
""", unsafe_allow_html=True)
# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
    --phosphor: #c8956c;
    --phosphor-dim: #a0724a;
    --phosphor-glow: rgba(200,149,108,0.15);
    --amber: #e07b39;
    --amber-dim: #b5612a;
    --silicon: #8b5e3c;
    --fab-blue: #c4a882;
    --bg-deep: #0f0b08;
    --bg-panel: #1a1208;
    --bg-card: #1f1610;
    --border: #2e2018;
    --border-bright: #3d2d1a;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: var(--bg-deep);
    background-image:
            linear-gradient(rgba(200,149,108,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(200,149,108,0.02) 1px, transparent 1px);
    background-size: 40px 40px;
    color: var(--text-primary);
}

/* circuit trace top bar */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #c8956c, #8b5e3c, #e07b39, transparent);
    z-index: 999;
}

section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border-bright) !important;
}

/* metric cards — IC pin label style */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-left: 3px solid var(--phosphor);
    border-radius: 4px;
    padding: 12px 16px;
    font-family: var(--mono);
}
[data-testid="stMetricValue"] {
    color: var(--phosphor) !important;
    font-family: var(--mono) !important;
    font-size: 1.8rem !important;
    font-weight: 400 !important;
    text-shadow: 0 0 20px rgba(0,255,136,0.4);
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,255,136,0.1) !important;
    color: var(--phosphor) !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
}

/* buttons */
.stButton > button {
    background: transparent;
    color: var(--phosphor);
    border: 1px solid var(--phosphor);
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.9rem;
    letter-spacing: 0.1em;
    padding: 10px 28px;
    width: 100%;
    text-transform: uppercase;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: rgba(0,255,136,0.1);
    box-shadow: 0 0 20px rgba(0,255,136,0.2);
}

/* headers */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
h1 { color: var(--text-primary) !important; }
h2 { color: var(--phosphor) !important; font-size: 1rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
h3 { color: var(--fab-blue) !important; font-size: 0.9rem !important; }

/* result cards */
.result-pass {
    background: var(--bg-card);
    border: 1px solid var(--phosphor);
    border-radius: 4px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,255,136,0.15), inset 0 0 40px rgba(0,255,136,0.03);
}
.result-fail {
    background: var(--bg-card);
    border: 1px solid #ef4444;
    border-radius: 4px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(239,68,68,0.15), inset 0 0 40px rgba(239,68,68,0.03);
}
.result-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--text-secondary);
    margin-bottom: 8px;
    text-transform: uppercase;
}
.result-value-pass { color: var(--phosphor); font-family: var(--mono); font-size: 3rem; text-shadow: 0 0 30px rgba(0,255,136,0.5); }
.result-value-fail { color: #ef4444; font-family: var(--mono); font-size: 3rem; text-shadow: 0 0 30px rgba(239,68,68,0.5); }
.result-conf { font-family: var(--mono); font-size: 0.9rem; margin-top: 12px; color: var(--text-secondary); }

/* chip pin card */
.chip-card {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-family: var(--mono);
    position: relative;
}
.chip-card::before {
    content: '';
    position: absolute;
    left: -6px; top: 50%; transform: translateY(-50%);
    width: 5px; height: 20px;
    background: var(--border-bright);
    border-radius: 1px 0 0 1px;
}
.chip-pin { color: var(--amber); font-size: 0.65rem; letter-spacing: 0.15em; margin-bottom: 4px; }
.chip-value { color: var(--text-primary); font-size: 0.9rem; }

.section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 16px;
}

hr { border-color: var(--border) !important; }

/* scanline overlay on charts */
.chart-wrap {
    border: 1px solid var(--border-bright);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
            /* SYSTEM ONLINE pulse dot */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px #00ff88; }
    50% { opacity: 0.3; box-shadow: none; }
}
.stApp [style*="SYSTEM ONLINE"] {
    animation: pulse-dot 1.8s ease-in-out infinite;
}

/* metric cards fade-in */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="metric-container"] {
    animation: fadeSlideUp 0.6s ease forwards;
}

/* predict button glow pulse */
@keyframes btn-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(200,149,108,0.3); }
    50%       { box-shadow: 0 0 24px rgba(200,149,108,0.7), 0 0 48px rgba(200,149,108,0.3); }
}
.stButton > button {
    animation: btn-glow 2s ease-in-out infinite;
}

/* PASS/FAIL CRT flicker */
@keyframes crt-flicker {
    0%   { opacity: 1; }
    2%   { opacity: 0.6; }
    4%   { opacity: 1; }
    8%   { opacity: 0.8; }
    10%  { opacity: 1; }
    92%  { opacity: 1; }
    94%  { opacity: 0.7; }
    96%  { opacity: 1; }
}
.result-value-pass, .result-value-fail {
    animation: crt-flicker 4s step-end infinite;
}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB THEME ──────────────────────────────────────────────────────────
PHOSPHOR = '#c8956c'
AMBER    = '#e07b39'
SILICON  = '#8b5e3c'
FAB_BLUE = '#c4a882'
RED_SIG  = '#c0392b'
MONO_BG  = '#0f0b08'
PANEL_BG = '#1a1208'
CARD_BG  = '#1f1610'
BORDER   = '#2e2018'

plt.rcParams.update({
    'figure.facecolor':  MONO_BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    '#1a2a4a',
    'axes.labelcolor':   '#64748b',
    'xtick.color':       '#475569',
    'ytick.color':       '#475569',
    'text.color':        '#e2e8f0',
    'grid.color':        '#0f1f30',
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
    'font.family':       'monospace',
})

# ── PIPELINE ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("signal-data.csv")
    total_columns = df.shape[1]
    sensor_columns = (["Time"] +
                      [f"Sensor_{i}" for i in range(1, total_columns - 1)] +
                      ["Pass/Fail"])
    df.columns = sensor_columns
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns.drop('Pass/Fail')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    Q1, Q3 = df[numeric_cols].quantile(0.25), df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_capped = df.copy()
    for col in numeric_cols:
        df_capped[col] = np.clip(df[col], Q1[col] - 1.5*IQR[col], Q3[col] + 1.5*IQR[col])
    class_counts = df['Pass/Fail'].value_counts()
    X = df_capped.drop(['Time','Pass/Fail'], axis=1)
    y = df['Pass/Fail']
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.25, random_state=42, stratify=y_res)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    return df, df_capped, X, y, X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, scaler, class_counts, numeric_cols

@st.cache_resource(show_spinner=False)
def train_models(_Xtr, _Xte, _ytr, _yte, _Xtr_raw, _Xte_raw):
    models = {}
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(_Xtr, _ytr); yp = lr.predict(_Xte)
    models["Logistic Regression"] = {"model":lr,"acc":accuracy_score(_yte,yp),"cm":confusion_matrix(_yte,yp),"cr":classification_report(_yte,yp,output_dict=True),"scaled":True}
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(_Xtr_raw, _ytr); yp = rf.predict(_Xte_raw)
    models["Random Forest"] = {"model":rf,"acc":accuracy_score(_yte,yp),"cm":confusion_matrix(_yte,yp),"cr":classification_report(_yte,yp,output_dict=True),"scaled":False}
    svc = SVC(C=1, kernel='rbf', probability=True, random_state=42)
    svc.fit(_Xtr, _ytr); yp = svc.predict(_Xte)
    models["SVM · RBF"] = {"model":svc,"acc":accuracy_score(_yte,yp),"cm":confusion_matrix(_yte,yp),"cr":classification_report(_yte,yp,output_dict=True),"scaled":True}
    return models

# ── WAFER MAP ─────────────────────────────────────────────────────────────────
def draw_wafer_map(y_series, title="Wafer Yield Map", size=12):
    n = min(len(y_series), size*size)
    vals = y_series.values[:n]
    fig, ax = plt.subplots(figsize=(4, 4))
    wafer_r = size / 2
    cx, cy = size / 2, size / 2
    theta = np.linspace(0, 2*np.pi, 200)
    ax.fill(cx + wafer_r*np.cos(theta), cy + wafer_r*np.sin(theta), color='#0d1628', zorder=0)
    ax.plot(cx + wafer_r*np.cos(theta), cy + wafer_r*np.sin(theta), color='#334155', lw=1.5, zorder=1)
    ax.plot(cx + (wafer_r*0.92)*np.cos(theta), cy + (wafer_r*0.92)*np.sin(theta), color='#1e3a5f', lw=0.5, ls='--', zorder=1)
    idx = 0
    for row in range(size):
        for col in range(size):
            x, y_pos = col + 0.5, row + 0.5
            dist = np.sqrt((x - cx)**2 + (y_pos - cy)**2)
            if dist < wafer_r * 0.91:
                if idx < len(vals):
                    color = PHOSPHOR if vals[idx] == -1 else RED_SIG
                    alpha = 0.85
                else:
                    color = '#1a2a3a'
                    alpha = 0.4
                die = plt.Rectangle((col+0.08, row+0.08), 0.84, 0.84,
                                     facecolor=color, edgecolor='#060a0f',
                                     linewidth=0.4, alpha=alpha, zorder=2)
                ax.add_patch(die)
                idx += 1
    flat_x = cx + wafer_r * np.cos(np.radians(270))
    ax.plot([flat_x - 0.8, flat_x + 0.8], [cy + wafer_r*np.sin(np.radians(270))]*2, color='#334155', lw=3, zorder=3)
    ax.set_xlim(0, size); ax.set_ylim(0, size)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, color='#94a3b8', fontsize=9, fontfamily='monospace', pad=6)
    pass_patch = mpatches.Patch(color=PHOSPHOR, label=f'Pass')
    fail_patch = mpatches.Patch(color=RED_SIG, label=f'Fail')
    ax.legend(handles=[pass_patch, fail_patch], loc='lower right',
              facecolor='#0a1020', edgecolor='#1e3a5f',
              labelcolor='#94a3b8', fontsize=7, framealpha=0.9)
    fig.patch.set_facecolor(MONO_BG)
    plt.tight_layout(pad=0.5)
    return fig

# ── LOAD ──────────────────────────────────────────────────────────────────────
try:
    with st.spinner("INITIALIZING YIELD ANALYSIS SYSTEM..."):
        df, df_capped, X, y, X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, scaler, class_counts, numeric_cols = load_data()
        models = train_models(X_tr_sc, X_te_sc, y_train, y_test, X_train, X_test)
        best_model = models["SVM · RBF"]["model"]
        data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px">
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#334155; letter-spacing:0.2em; margin-bottom:4px">SEMIYIELD · v2.1.0</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; color:#e2e8f0; font-weight:600">Wafer Intelligence<br>System</div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#00ff88; margin-top:6px">● SYSTEM ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if data_loaded:
        n_pass = class_counts.get(-1, 0)
        n_fail = class_counts.get(1, 0)
        yield_pct = n_pass / (n_pass + n_fail) * 100

        st.markdown('<div class="section-label">// fab metrics</div>', unsafe_allow_html=True)

        for label, val in [
            ("WAFER SAMPLES", f"{len(df):,}"),
            ("SENSOR CHANNELS", f"{X.shape[1]}"),
            ("RAW YIELD RATE", f"{yield_pct:.1f}%"),
            ("BEST MODEL ACC", f"{models['SVM · RBF']['acc']:.4f}"),
        ]:
            st.markdown(f"""
            <div class="chip-card">
                <div class="chip-pin">PIN · {label}</div>
                <div class="chip-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">// pipeline</div>', unsafe_allow_html=True)
        steps = ["CSV Ingestion","Median Imputation","IQR Capping","SMOTE Balance","Train/Test Split","StandardScaler","GridSearchCV","SVM · RBF"]
        for i, s in enumerate(steps):
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#334155; padding:3px 0">
                <span style="color:#1e3a5f">{i:02d}</span>
                <span style="color:#00ff88; margin:0 6px">▶</span>
                <span style="color:#64748b">{s}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#334155; text-align:center; line-height:2">
        BUILT BY <span style="color:#64748b">SHREYA</span><br>
        CS ENGINEERING · AI/ML<br>
        <a href="https://github.com/12Shreya8" style="color:#0ea5e9">GITHUB</a>
        <span style="color:#1e3a5f"> · </span>
        <a href="https://linkedin.com/in/shreya-yergol" style="color:#0ea5e9">LINKEDIN</a>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
if not data_loaded:
    st.error("signal-data.csv not found. Place it in the same directory as app.py.")
    st.stop()

# Hero
st.markdown("""
<div style="padding:24px 0 20px">
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#334155; letter-spacing:0.2em; margin-bottom:8px">
        ⬡ SEMICONDUCTOR MANUFACTURING YIELD PREDICTION · SECOM DATASET
    </div>
    <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.2rem; margin:0; color:#e2e8f0; font-weight:700">
        SemiYield Intelligence Dashboard
    </h1>
    <p style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#475569; margin-top:8px">
        590+ sensor channels · SMOTE balancing · SVM with RBF kernel · real-time die prediction
    </p>
</div>
""", unsafe_allow_html=True)

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
n_pass = class_counts.get(-1,0); n_fail = class_counts.get(1,0)
with c1: st.metric("TOTAL DIES", f"{len(df):,}")
with c2: st.metric("SENSORS", f"{X.shape[1]}")
with c3: st.metric("YIELD RATE", f"{n_pass/(n_pass+n_fail)*100:.1f}%")
with c4: st.metric("SVM ACC", f"{models['SVM · RBF']['acc']:.4f}")
with c5: st.metric("FAIL RATE", f"{n_fail/(n_pass+n_fail)*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⬡  WAFER · EDA", "◈  MODEL · RESULTS", "◉  LIVE · PREDICTOR"])

# ══════════════════════════════════════════════════
# TAB 1 · EDA
# ══════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Wafer Yield Map")
        st.caption("Die-level pass/fail visualization — simulated wafer grid from dataset")
        fig = draw_wafer_map(y, "RAW DATASET · BEFORE SMOTE", size=14)
        st.pyplot(fig)

    with col_right:
        st.markdown("### Class Imbalance")
        st.caption("Target distribution before and after SMOTE oversampling")
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.5))

        # Before
        axes[0].bar(['Pass','Fail'], [n_pass, n_fail],
                    color=[PHOSPHOR, RED_SIG], edgecolor=MONO_BG, linewidth=1, width=0.5)
        axes[0].set_title("Before SMOTE", color='#94a3b8', fontsize=9)
        axes[0].set_facecolor(PANEL_BG)
        for spine in axes[0].spines.values(): spine.set_color(BORDER)
        axes[0].grid(axis='y', alpha=0.3)

        # After
        after = [sum(y_train==-1)+sum(y_test==-1), sum(y_train==1)+sum(y_test==1)]
        axes[1].bar(['Pass','Fail'], after,
                    color=[PHOSPHOR, RED_SIG], edgecolor=MONO_BG, linewidth=1, width=0.5)
        axes[1].set_title("After SMOTE", color='#94a3b8', fontsize=9)
        axes[1].set_facecolor(PANEL_BG)
        for spine in axes[1].spines.values(): spine.set_color(BORDER)
        axes[1].grid(axis='y', alpha=0.3)

        for ax in axes:
            for bar in ax.patches:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                        str(int(bar.get_height())), ha='center', va='bottom',
                        fontsize=8, color='#94a3b8', fontfamily='monospace')
        fig.patch.set_facecolor(MONO_BG)
        plt.tight_layout(pad=1)
        st.pyplot(fig)

    st.markdown("---")

    st.markdown("### Outlier Treatment · IQR Capping")
    sample_cols = list(numeric_cols[:8])
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    for idx, (data, label, col) in enumerate([
        (df[sample_cols], "BEFORE CAPPING", AMBER),
        (df_capped[sample_cols], "AFTER CAPPING", PHOSPHOR)
    ]):
        bp = data.boxplot(ax=axes[idx], patch_artist=True, return_type='dict',
                          boxprops=dict(facecolor=f'{col}22', color=col, linewidth=0.8),
                          medianprops=dict(color=FAB_BLUE, linewidth=2),
                          whiskerprops=dict(color='#334155', linewidth=0.8),
                          capprops=dict(color='#334155', linewidth=0.8),
                          flierprops=dict(markerfacecolor=RED_SIG, marker='.', markersize=2, alpha=0.4))
        axes[idx].set_title(label, color='#94a3b8', fontsize=9)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=7)
        axes[idx].set_facecolor(PANEL_BG)
        for spine in axes[idx].spines.values(): spine.set_color(BORDER)
        axes[idx].grid(axis='y', alpha=0.3)
    fig.patch.set_facecolor(MONO_BG)
    plt.tight_layout(pad=1)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Sensor Correlation · Top 15 Channels")
    top15 = list(numeric_cols[:15])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    corr = df_capped[top15].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    custom_cmap = sns.diverging_palette(145, 280, s=85, l=25, as_cmap=True)
    sns.heatmap(corr, ax=ax, cmap='RdYlGn', center=0, mask=mask,
                annot=True, fmt='.1f', annot_kws={"size":7, "color":"#1a0f05"},
                linewidths=0.5, linecolor=MONO_BG,
                cbar_kws={'shrink':0.8, 'label':'Correlation'})
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors='#475569', labelsize=8)
    fig.patch.set_facecolor(MONO_BG)
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════
# TAB 2 · MODEL RESULTS
# ══════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Accuracy · Signal Comparison")

    m_names = list(models.keys())
    m_accs  = [models[m]["acc"] for m in m_names]
    bar_colors = [FAB_BLUE, SILICON, PHOSPHOR]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(m_names, m_accs, color=bar_colors, edgecolor=MONO_BG, linewidth=1.5, width=0.45)
    ax.set_ylim(0.88, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("MODEL PERFORMANCE COMPARISON", color='#475569', fontsize=9, fontfamily='monospace')
    for bar, acc in zip(bars, m_accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{acc:.4f}", ha='center', color='#e2e8f0', fontsize=10, fontfamily='monospace')
    for spine in ax.spines.values(): spine.set_color(BORDER)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.9, color=AMBER, lw=0.8, ls='--', alpha=0.5, label='0.90 baseline')
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor='#64748b')
    fig.patch.set_facecolor(MONO_BG)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Confusion Matrices · All Models")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for idx, (name, info) in enumerate(models.items()):
        cm = info["cm"]
        cmap = sns.light_palette(bar_colors[idx], as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                    cmap=cmap, linewidths=1.5, linecolor=MONO_BG,
                    annot_kws={"size":14, "weight":"bold", "color":"#1a0f05"})
        axes[idx].set_title(f"{name} · {info['acc']:.4f}", color='#94a3b8', fontsize=9, fontfamily='monospace')
        axes[idx].set_xlabel("PREDICTED", color='#475569', fontsize=8)
        axes[idx].set_ylabel("ACTUAL", color='#475569', fontsize=8)
        axes[idx].set_xticklabels(['Pass','Fail'], color='#64748b', fontsize=8)
        axes[idx].set_yticklabels(['Pass','Fail'], color='#64748b', fontsize=8, rotation=0)
        axes[idx].set_facecolor(PANEL_BG)
    fig.patch.set_facecolor(MONO_BG)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Classification Report")

    sel = st.selectbox("SELECT MODEL", list(models.keys()),
                       format_func=lambda x: f"◈ {x}")
    cr = models[sel]["cr"]
    rows = []
    for lk, ln in [("-1","Pass"),("1","Fail")]:
        if lk in cr:
            r = cr[lk]
            rows.append({"CLASS":ln,"PRECISION":f"{r['precision']:.4f}",
                         "RECALL":f"{r['recall']:.4f}","F1-SCORE":f"{r['f1-score']:.4f}",
                         "SUPPORT":int(r['support'])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # radar-style metric comparison
    st.markdown("### SVM vs Random Forest · Key Metrics")
    fig, ax = plt.subplots(figsize=(9, 3))
    metric_labels = ['Precision\n(Pass)', 'Recall\n(Pass)', 'F1\n(Pass)',
                     'Precision\n(Fail)', 'Recall\n(Fail)', 'F1\n(Fail)']
    x = np.arange(len(metric_labels))
    w = 0.28
    for i, (mname, col) in enumerate([("SVM · RBF", PHOSPHOR), ("Random Forest", SILICON)]):
        cr_ = models[mname]["cr"]
        vals = [cr_["-1"]["precision"], cr_["-1"]["recall"], cr_["-1"]["f1-score"],
                cr_["1"]["precision"], cr_["1"]["recall"], cr_["1"]["f1-score"]]
        ax.bar(x + i*w, vals, w, label=mname, color=col, edgecolor=MONO_BG, alpha=0.85)
    ax.set_xticks(x + w/2); ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_ylim(0.8, 1.02); ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor='#94a3b8')
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values(): spine.set_color(BORDER)
    fig.patch.set_facecolor(MONO_BG)
    ax.set_title("PRECISION · RECALL · F1 BREAKDOWN", color='#475569', fontsize=8, fontfamily='monospace')
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════
# TAB 3 · LIVE PREDICTOR
# ══════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#334155; padding:12px 0 4px; letter-spacing:0.1em">
        // REAL-TIME DIE CLASSIFICATION · SVM · RBF KERNEL · ADJUST SENSOR READINGS BELOW
    </div>
    """, unsafe_allow_html=True)

    top_sensors = X.var().nlargest(10).index.tolist()
    stats = df_capped[X.columns].describe()
    sample_row = X.iloc[0]
    user_input = dict(zip(X.columns, sample_row.values))

    st.markdown("#### Top 10 Sensors by Variance")
    cols = st.columns(2)
    for idx, sensor in enumerate(top_sensors):
        mn = float(stats[sensor]['min'])
        mx = float(stats[sensor]['max'])
        mean_v = float(stats[sensor]['mean'])
        default = float(sample_row[sensor])
        with cols[idx % 2]:
            user_input[sensor] = st.slider(
                f"⬡ {sensor}",
                min_value=round(mn, 2), max_value=round(mx, 2),
                value=round(default, 2),
                step=round((mx-mn)/100, 4),
                help=f"μ={mean_v:.2f}  range=[{mn:.1f}, {mx:.1f}]"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("◉  RUN YIELD PREDICTION"):
        input_arr = np.array([[user_input[c] for c in X.columns]])
        input_sc  = scaler.transform(input_arr)
        pred      = best_model.predict(input_sc)[0]
        proba     = best_model.predict_proba(input_sc)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1.2, 1])

        with left:
            if pred == -1:
                conf = proba[0]*100
                st.markdown(f"""
                <div class="result-pass">
                    <div class="result-label">// yield classification output</div>
                    <div class="result-value-pass">PASS</div>
                    <div class="result-conf">CONFIDENCE · {conf:.1f}%</div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#334155; margin-top:16px">
                        STATUS · DIE CLEARED FOR PACKAGING
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                conf = proba[1]*100
                st.markdown(f"""
                <div class="result-fail">
                    <div class="result-label">// yield classification output</div>
                    <div class="result-value-fail">FAIL</div>
                    <div class="result-conf">CONFIDENCE · {conf:.1f}%</div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#334155; margin-top:16px">
                        STATUS · DIE FLAGGED FOR INSPECTION
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with right:
            # probability oscilloscope-style bar
            fig, ax = plt.subplots(figsize=(4, 3.5))
            probs = [proba[0], proba[1]]
            colors = [PHOSPHOR, RED_SIG]
            bars = ax.barh(['PASS','FAIL'], probs, color=colors, edgecolor=MONO_BG, height=0.4)
            ax.set_xlim(0, 1.1)
            ax.set_title("PROBABILITY OUTPUT", color='#475569', fontsize=8, fontfamily='monospace')
            for bar, p in zip(bars, probs):
                ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                        f"{p*100:.1f}%", va='center', color='#94a3b8', fontsize=10, fontfamily='monospace')
            ax.axvline(x=0.5, color=AMBER, lw=1, ls='--', alpha=0.5)
            ax.text(0.5, 1.05, "50%", ha='center', color=AMBER, fontsize=7, fontfamily='monospace', transform=ax.get_xaxis_transform())
            ax.set_facecolor(PANEL_BG)
            for spine in ax.spines.values(): spine.set_color(BORDER)
            ax.tick_params(colors='#475569', labelsize=8)
            fig.patch.set_facecolor(MONO_BG)
            plt.tight_layout()
            st.pyplot(fig)

        # mini wafer map showing prediction
        st.markdown("---")
        st.markdown("#### Simulated Wafer Impact")
        st.caption("Wafer map with this die's predicted classification highlighted")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig = draw_wafer_map(y, "FULL DATASET YIELD MAP", size=14)
            st.pyplot(fig)
        with col_w2:
            synthetic_y = pd.Series([-1]*120 + [pred]*1 + [-1]*23)
            fig = draw_wafer_map(synthetic_y, "CURRENT PREDICTION (highlighted)", size=12)
            st.pyplot(fig)
