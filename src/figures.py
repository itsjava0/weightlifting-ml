"""The report figures. Reads only files produced by the other
scripts, so every figure matches the numbers in reports/.

Following APA style, figures carry no title inside the image: the figure
number, title and note are written in the report above and below each one.

Outputs (in reports/figures/):
  fig1_progression.png       Lat Pulldown e1RM over time, gym change marked
  fig2_pred_vs_actual.png    predicted vs actual, baseline and Ridge
  fig3_error_by_exercise.png baseline vs Ridge MAE per exercise
  fig4_ablation.png          feature test: MAE of every run
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "figures"
OUT.mkdir(exist_ok=True)

# Palette (validated reference palette, light mode) and recessive chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT / name}")


# --- Figure 1: progression of one exercise ----------------------------------
df = pd.read_csv(ROOT / "data" / "processed" / "model_dataset.csv", parse_dates=["timestamp"])
ex = df[df["exercise"] == "Lat Pulldown"].sort_values("timestamp")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(ex["timestamp"], ex["performance_score"], color=BLUE, lw=2, zorder=2)
ax.scatter(ex["timestamp"], ex["performance_score"], s=22, color=BLUE,
           edgecolor="white", linewidth=1, zorder=3)
gym = pd.Timestamp("2025-10-01")
ax.axvline(gym, color=INK_3, lw=1, ls="--", zorder=1)
ax.text(gym, ax.get_ylim()[1], "  rotating 3-4 gyms from here", va="top",
        ha="left", color=INK_2, fontsize=9)
ax.set_ylabel("e1RM (kg)")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.grid(axis="x", visible=False)
save(fig, "fig1_progression.png")

# --- Figure 2: predicted vs actual -----------------------------------------
base = pd.read_csv(REPORTS / "baseline_predictions.csv")
base = base[base["method"] == "last_value"]
ridge = pd.read_csv(REPORTS / "ridge_predictions.csv")
lo = min(base[["actual", "predicted"]].min().min(), ridge["predicted"].min()) - 5
hi = max(base[["actual", "predicted"]].max().max(), ridge["predicted"].max()) + 5
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharex=True, sharey=True)
for ax, p, color, label in [(axes[0], base, BLUE, "Baseline (same as last session)"),
                            (axes[1], ridge, ORANGE, "Ridge regression")]:
    ax.plot([lo, hi], [lo, hi], color=INK_3, lw=1, ls="--", zorder=1)
    ax.scatter(p["actual"], p["predicted"], s=18, color=color, alpha=0.75,
               edgecolor="white", linewidth=0.6, zorder=2)
    mae = (p["predicted"] - p["actual"]).abs().mean()
    ax.set_title(label, fontsize=11)
    ax.text(0.04, 0.96, f"MAE {mae:.2f} kg", transform=ax.transAxes,
            va="top", color=INK, fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual next-session e1RM (kg)")
axes[0].set_ylabel("Predicted e1RM (kg)")
fig.tight_layout()
save(fig, "fig2_pred_vs_actual.png")

# --- Figure 3: error per exercise ------------------------------------------
t = pd.read_csv(REPORTS / "ridge_vs_baseline.csv")
t = t[t["exercise"] != "ALL"].sort_values("baseline_MAE")
y = range(len(t))
h = 0.38
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.barh([i + h / 2 + 0.02 for i in y], t["baseline_MAE"], height=h, color=BLUE,
        label="Baseline (same as last session)")
ax.barh([i - h / 2 - 0.02 for i in y], t["ridge_MAE"], height=h, color=ORANGE,
        label="Ridge regression")
for i, (b, r) in enumerate(zip(t["baseline_MAE"], t["ridge_MAE"])):
    ax.text(b + 0.08, i + h / 2 + 0.02, f"{b:.2f}", va="center", fontsize=8, color=INK_2)
    ax.text(r + 0.08, i - h / 2 - 0.02, f"{r:.2f}", va="center", fontsize=8, color=INK_2)
ax.set_yticks(list(y))
ax.set_yticklabels(t["exercise"])
ax.set_xlabel("Mean absolute error (kg), lower is better")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
save(fig, "fig3_error_by_exercise.png")

# --- Figure 4: ablation ------------------------------------------------------
a = pd.read_csv(REPORTS / "ablation_results.csv")
a = a[~a["run"].str.startswith("drop ")].copy()
a["label"] = (a["run"].str.replace("only ", "only: ", regex=False)
              .str.replace("full model (all 6)", "all 6 features", regex=False)
              .str.replace("no features (drift only)", "no features (average change)", regex=False))
a = a.sort_values("MAE", ascending=False)
base_mae = a["MAE"].iloc[0] - a["vs_baseline"].iloc[0]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(a["label"], a["MAE"], height=0.6, color=ORANGE)
for i, v in enumerate(a["MAE"]):
    ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
ax.axvline(base_mae, color=BLUE, lw=2)
ax.text(base_mae - 0.02, -0.75, f"baseline {base_mae:.2f}", color=INK,
        ha="right", va="center", fontsize=9)
ax.set_ylim(-1.1, len(a) - 0.5)
ax.set_xlim(3.5, a["MAE"].max() + 0.2)
ax.set_xlabel("Mean absolute error (kg), lower is better (axis starts at 3.5)")
ax.grid(axis="y", visible=False)
save(fig, "fig4_ablation.png")
