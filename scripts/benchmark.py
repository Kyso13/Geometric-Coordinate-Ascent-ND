"""
=================================================================
  GCA-ND (C++)  vs  sklearn OLS  —  Benchmark & Görsel
  3 Senaryo: OLS'nin yetersiz kaldığı durumlar
=================================================================
"""

import numpy as np
import subprocess, time, os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
warnings.filterwarnings("ignore")

GCA_BIN = "/home/claude/gca_nd"

# ── C++ GCA çağırıcısı ─────────────────────────────────────
def run_gca(X, y, loss_id=0, lam=0.01):
    m, n = X.shape
    lines = [f"{m} {n}"]
    for i in range(m):
        row = " ".join(f"{X[i,j]:.8f}" for j in range(n))
        lines.append(f"{row} {y[i]:.8f}")
    data_str = "\n".join(lines)

    t0 = time.perf_counter()
    result = subprocess.run(
        [GCA_BIN, str(loss_id), str(lam)],
        input=data_str, capture_output=True, text=True
    )
    elapsed = (time.perf_counter() - t0) * 1000  # ms

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    lines_out = result.stdout.strip().split("\n")
    vals   = list(map(float, lines_out[0].split()))
    theta  = np.array(vals[:n+1])
    mse, mae, r2, cpp_ms, iters = vals[n+1], vals[n+2], vals[n+3], vals[n+4], int(vals[n+5])
    f_hist = list(map(float, lines_out[1].split()))
    return {"theta": theta, "mse": mse, "mae": mae, "r2": r2,
            "time_ms": elapsed, "cpp_ms": cpp_ms, "iters": iters,
            "f_hist": f_hist}

# ══════════════════════════════════════════════════════════
# SENARYO TANIMLARI
# ══════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

def make_scenario_1(n=200):
    """Aşırı Aykırı Değerler — OLS patlıyor, MAE/Huber GCA dayanıyor"""
    X = rng.standard_normal((n, 3))
    y = 4*X[:,0] - 2*X[:,1] + 3*X[:,2] + 5 + rng.normal(0, 0.5, n)
    # %15 oranında çok güçlü aykırı değer
    n_out = int(n * 0.20)
    idx = rng.choice(n, n_out, replace=False)
    y[idx] += 45.0   # tek yönlü, OLS maxi çeker
    true = np.array([5, 4, -2, 3])
    return X, y, true

def make_scenario_2(n=150):
    """L1 Regularization (Lasso) — Seyrek veri, OLS overfit, GCA-Lasso seyrek çözüm"""
    n_feat = 8
    X = rng.standard_normal((n, n_feat))
    # Sadece 3 özellik gerçekten ilgili
    true_w = np.array([3.0, -2.5, 4.0, 0, 0, 0, 0, 0])
    y = 6 + X @ true_w + rng.normal(0, 1.2, n)
    true = np.concatenate([[6], true_w])
    return X, y, true

def make_scenario_3(n=150):
    """Non-Gaussian Gürültü (Laplace) — OLS varsayımı ihlali, MAE GCA daha iyi"""
    X = rng.standard_normal((n, 4))
    y = 3*X[:,0] + 1.5*X[:,1] - 2*X[:,2] + X[:,3] + 7
    # Laplace gürültüsü (kalın kuyruk)
    y += rng.laplace(0, 2.5, n)
    true = np.array([7, 3, 1.5, -2, 1])
    return X, y, true

SCENARIOS = [
    {"name": "Senaryo 1: Aykırı Değer Saldırısı",
     "subtitle": "%15 outlier  |  GCA-MAE vs OLS",
     "make": make_scenario_1,
     "gca_loss": 1,   # MAE
     "gca_label": "GCA-ND (MAE)",
     "ols_label": "sklearn OLS",
     "ols_cls": LinearRegression,
     "ols_kw": {},
     "ref_label": "sklearn Huber",
     "ref_cls": HuberRegressor,
     "ref_kw": {"epsilon": 1.35},
    },
    {"name": "Senaryo 2: Seyrek Veri (Lasso)",
     "subtitle": "8 özellik, 3 gerçek  |  GCA-L1 vs OLS",
     "make": make_scenario_2,
     "gca_loss": 3,   # LASSO
     "gca_label": "GCA-ND (L1)",
     "ols_label": "sklearn OLS",
     "ols_cls": LinearRegression,
     "ols_kw": {},
     "ref_label": "sklearn Lasso",
     "ref_cls": Lasso,
     "ref_kw": {"alpha": 0.1},
    },
    {"name": "Senaryo 3: Laplace Gürültüsü",
     "subtitle": "Kalın kuyruklu gürültü  |  GCA-Huber vs OLS",
     "make": make_scenario_3,
     "gca_loss": 2,   # HUBER
     "gca_label": "GCA-ND (Huber)",
     "ols_label": "sklearn OLS",
     "ols_cls": LinearRegression,
     "ols_kw": {},
     "ref_label": "sklearn Huber",
     "ref_cls": HuberRegressor,
     "ref_kw": {"epsilon": 1.35},
    },
]

# ──── Her senaryoyu çalıştır ───────────────────────────────
results = []
for sc in SCENARIOS:
    X, y, true = sc["make"]()
    n_feat = X.shape[1]

    # GCA-ND (C++)
    gca = run_gca(X, y, loss_id=sc["gca_loss"], lam=0.05)

    # OLS
    t0 = time.perf_counter()
    ols_m = sc["ols_cls"](**sc["ols_kw"]).fit(X, y)
    ols_t = (time.perf_counter()-t0)*1000
    if hasattr(ols_m, "coef_"):
        ols_pred = ols_m.predict(X)
    else:
        ols_pred = ols_m.predict(X)

    # Ref (sklearn Huber / Lasso)
    t0 = time.perf_counter()
    ref_m = sc["ref_cls"](**sc["ref_kw"]).fit(X, y)
    ref_t = (time.perf_counter()-t0)*1000
    ref_pred = ref_m.predict(X)

    gca_pred = (np.hstack([np.ones((len(X),1)), X]) @ gca["theta"])

    results.append({
        "sc": sc, "X": X, "y": y, "true": true,
        "gca": gca, "gca_pred": gca_pred,
        "ols_pred": ols_pred, "ols_t": ols_t,
        "ref_pred": ref_pred, "ref_t": ref_t,
        "n_feat": n_feat,
    })
    print(f"[{sc['name']}]")
    print(f"  GCA  R²={r2_score(y,gca_pred):.4f}  MAE={mean_absolute_error(y,gca_pred):.3f}  t={gca['time_ms']:.1f}ms")
    print(f"  OLS  R²={r2_score(y,ols_pred):.4f}  MAE={mean_absolute_error(y,ols_pred):.3f}  t={ols_t:.2f}ms")
    print(f"  REF  R²={r2_score(y,ref_pred):.4f}  MAE={mean_absolute_error(y,ref_pred):.3f}  t={ref_t:.2f}ms\n")

# ══════════════════════════════════════════════════════════
# VİZÜALİZASYON
# ══════════════════════════════════════════════════════════
BG    = "#07080f"
PANEL = "#0d1120"
TEXT  = "#dce8f5"
GRID  = "#151e30"
BLUE  = "#38bdf8"
ORG   = "#fb923c"
PURPLE= "#a78bfa"
GOLD  = "#fbbf24"
GREEN = "#4ade80"
RED   = "#f87171"

fig = plt.figure(figsize=(22, 17), facecolor=BG)
fig.suptitle(
    "GCA-ND (C++, Gradyansız)  vs  sklearn OLS  —  3 Kritik Senaryo",
    fontsize=15, color=TEXT, fontweight="bold", y=0.982, family="monospace"
)

# 3 satır × 4 sütun grid
# Her satır: parity_gca | parity_ols | bar_mae | coefficients
outer_gs = gridspec.GridSpec(4, 1, figure=fig,
    left=0.04, right=0.98, top=0.95, bottom=0.04,
    hspace=0.55)

def style(ax, title, fs=8.5):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=fs, pad=6, family="monospace")
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.grid(True, color=GRID, alpha=0.7, lw=0.55)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

colors_method = [BLUE, ORG, PURPLE]
method_labels = None   # senaryo bazlı

for row_i, r in enumerate(results):
    sc = r["sc"]
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer_gs[row_i],
        wspace=0.35
    )

    gca_r2  = r2_score(r["y"], r["gca_pred"])
    ols_r2  = r2_score(r["y"], r["ols_pred"])
    ref_r2  = r2_score(r["y"], r["ref_pred"])
    gca_mae = mean_absolute_error(r["y"], r["gca_pred"])
    ols_mae = mean_absolute_error(r["y"], r["ols_pred"])
    ref_mae = mean_absolute_error(r["y"], r["ref_pred"])

    # ── Parity: GCA ──────────────────────────────────────
    ax0 = fig.add_subplot(inner[0])
    style(ax0, f"{sc['name']}\nParity: {sc['gca_label']}")
    ax0.set_xlabel("Gerçek y", fontsize=7.5)
    ax0.set_ylabel("Tahmin ŷ", fontsize=7.5)
    mn = min(r["y"].min(), r["gca_pred"].min())
    mx = max(r["y"].max(), r["gca_pred"].max())
    pad = (mx-mn)*0.06
    d = np.linspace(mn-pad, mx+pad, 100)
    ax0.scatter(r["y"], r["gca_pred"], color=BLUE, alpha=0.5, s=18)
    ax0.plot(d, d, color=GOLD, lw=1.5, ls="--")
    ax0.text(0.05, 0.90, f"R²={gca_r2:.4f}\nMAE={gca_mae:.3f}",
             transform=ax0.transAxes, color=GREEN, fontsize=8,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=GRID, alpha=0.8))
    ax0.set_xlim(mn-pad, mx+pad); ax0.set_ylim(mn-pad, mx+pad); ax0.set_aspect("equal")

    # ── Parity: OLS ──────────────────────────────────────
    ax1 = fig.add_subplot(inner[1])
    style(ax1, f"{sc['subtitle']}\nParity: {sc['ols_label']}")
    ax1.set_xlabel("Gerçek y", fontsize=7.5)
    ax1.set_ylabel("Tahmin ŷ", fontsize=7.5)
    mn2 = min(r["y"].min(), r["ols_pred"].min())
    mx2 = max(r["y"].max(), r["ols_pred"].max())
    pad2 = (mx2-mn2)*0.06
    d2 = np.linspace(mn2-pad2, mx2+pad2, 100)
    ax1.scatter(r["y"], r["ols_pred"], color=ORG, alpha=0.5, s=18)
    ax1.plot(d2, d2, color=GOLD, lw=1.5, ls="--")
    ax1.text(0.05, 0.90, f"R²={ols_r2:.4f}\nMAE={ols_mae:.3f}",
             transform=ax1.transAxes, color=RED, fontsize=8,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=GRID, alpha=0.8))
    ax1.set_xlim(mn2-pad2, mx2+pad2); ax1.set_ylim(mn2-pad2, mx2+pad2); ax1.set_aspect("equal")

    # ── MAE Karşılaştırma Çubuğu ─────────────────────────
    ax2 = fig.add_subplot(inner[2])
    style(ax2, "MAE Karşılaştırması\n(düşük = iyi)")
    ax2.set_ylabel("MAE", fontsize=7.5)
    names  = [sc["gca_label"], sc["ols_label"], sc["ref_label"]]
    maes   = [gca_mae, ols_mae, ref_mae]
    bars   = ax2.bar(names, maes, color=[BLUE, ORG, PURPLE],
                     alpha=0.85, width=0.5, edgecolor=BG, linewidth=0.5)
    for bar, v in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(maes)*0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 color=TEXT, fontsize=8, family="monospace")
    ax2.tick_params(axis="x", labelsize=7, rotation=10)
    best_idx = np.argmin(maes)
    bars[best_idx].set_edgecolor(GREEN)
    bars[best_idx].set_linewidth(2)

    # ── Katsayı Karşılaştırması ───────────────────────────
    ax3 = fig.add_subplot(inner[3])
    true_all = r["true"]
    gca_all  = r["gca"]["theta"][:len(true_all)]
    n_show   = min(len(true_all), 6)   # maksimum 6 parametre göster
    style(ax3, f"Katsayılar (ilk {n_show})\n[Gerçek vs GCA vs OLS]")
    ax3.set_ylabel("Değer", fontsize=7.5)
    xp = np.arange(n_show); w = 0.26

    if hasattr(r["ols_pred"], "__len__"):
        ols_m_ref = sc["ols_cls"](**sc["ols_kw"]).fit(r["X"], r["y"])
        if hasattr(ols_m_ref, "intercept_"):
            ols_all = np.concatenate([[ols_m_ref.intercept_], ols_m_ref.coef_])
        else:
            ols_all = true_all * 0
    else:
        ols_all = true_all * 0

    ax3.bar(xp - w,   true_all[:n_show], w, color=GOLD,  alpha=0.85, label="Gerçek")
    ax3.bar(xp,       gca_all[:n_show],  w, color=BLUE,  alpha=0.85, label="GCA-ND")
    ax3.bar(xp + w,   ols_all[:n_show],  w, color=ORG,   alpha=0.85, label="OLS")
    ax3.axhline(0, color=TEXT, lw=0.4, alpha=0.3)
    ax3.set_xticks(xp)
    ax3.set_xticklabels([f"θ{i}" for i in range(n_show)], fontsize=7.5)
    ax3.legend(fontsize=6.5, facecolor=BG, labelcolor=TEXT,
               framealpha=0.7, loc="best")

# ── Son satır: Hız + Özet ─────────────────────────────────
bottom_inner = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[3], wspace=0.3
)

# Hız karşılaştırma
ax_spd = fig.add_subplot(bottom_inner[0])
style(ax_spd, "⑦ Çalışma Süresi Karşılaştırması  [ms, log ölçek]")
ax_spd.set_ylabel("Süre (ms)", fontsize=8)
ax_spd.set_yscale("log")

scenario_names = ["S1: Outlier\n(MAE)", "S2: Lasso\n(L1)", "S3: Laplace\n(Huber)"]
gca_times = [r["gca"]["time_ms"] for r in results]
ols_times = [r["ols_t"] for r in results]
ref_times = [r["ref_t"] for r in results]
x3 = np.arange(3); w3 = 0.25
ax_spd.bar(x3 - w3, gca_times, w3, color=BLUE, alpha=0.85, label="GCA-ND (C++)")
ax_spd.bar(x3,      ols_times, w3, color=ORG,  alpha=0.85, label="sklearn OLS")
ax_spd.bar(x3 + w3, ref_times, w3, color=PURPLE, alpha=0.85, label="sklearn Ref")
for i, (g,o) in enumerate(zip(gca_times, ols_times)):
    ax_spd.text(i - w3, g*1.15, f"{g:.0f}", ha="center",
                fontsize=7.5, color=TEXT, family="monospace")
    ax_spd.text(i, o*1.15, f"{o:.1f}", ha="center",
                fontsize=7.5, color=TEXT, family="monospace")
ax_spd.set_xticks(x3); ax_spd.set_xticklabels(scenario_names, fontsize=8)
ax_spd.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.7)

# Özet metin tablosu
ax_sum = fig.add_subplot(bottom_inner[1])
ax_sum.set_facecolor(PANEL)
ax_sum.set_title("⑧ Genel Özet", color=TEXT, fontsize=9, pad=6, family="monospace")
ax_sum.axis("off")

rows = [
    ("", "GCA-ND (C++)", "sklearn OLS", "Kazanan"),
    ("─"*10, "─"*12, "─"*12, "─"*8),
]
for i, r in enumerate(results):
    gca_mae_v = mean_absolute_error(r["y"], r["gca_pred"])
    ols_mae_v = mean_absolute_error(r["y"], r["ols_pred"])
    winner = "✓ GCA" if gca_mae_v < ols_mae_v else "✓ OLS"
    winner_c = GREEN if "GCA" in winner else RED
    rows.append((f"S{i+1} MAE",
                 f"{gca_mae_v:.3f}",
                 f"{ols_mae_v:.3f}",
                 winner))

rows += [
    ("─"*10, "─"*12, "─"*12, "─"*8),
    ("Gradyan?", "✗ YOK", "✓ VAR", ""),
    ("Loss seç.", "MSE/MAE/Huber", "Sadece MSE", ""),
    ("Reg.", "L1+L2", "Yok (OLS)", ""),
    ("Dil", "C++ (native)", "Python", ""),
]

col_x = [0.00, 0.28, 0.56, 0.81]
col_c = [TEXT, BLUE, ORG, GREEN]
yp = 0.97
for row in rows:
    for xi, (val, cc) in enumerate(zip(row, col_c)):
        if xi == 3 and "✓ OLS" in str(val):
            cc = RED
        ax_sum.text(col_x[xi], yp, str(val), transform=ax_sum.transAxes,
                    color=cc, fontsize=7.8, family="monospace",
                    verticalalignment="top")
    yp -= 0.082

out = "/mnt/user-data/outputs/gca_cpp_vs_ols_benchmark.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\nGörsel kaydedildi: {out}")
