"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_exposure.png, 02_effects.png, 03_age.png, 04_importance.png, 05_calibration.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
CE, CD, CF, IM, MG, SC = (pd.read_csv(R / f"{f}.csv") for f in ("calibration_exposure", "calibration_deciles", "coefficients", "importance", "marginal", "scores")); CF = CF.set_index("term")
AGE_LEVELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def exposure_share(ax, size=9):
    x = np.arange(len(CE)); ax.bar(x, 100 * CE["actual_claim_share"], color=GRAY, width=0.62, label="actual")
    ax.plot(x, 100 * CE["logistic_claim_share"], marker="o", color=BLUE, linewidth=2, label="logistic model, no exposure"); ax.plot(x, 100 * CE["poisson_claim_share"], marker="o", color=ORANGE, linewidth=2, label="Poisson model, with exposure")
    for xi, v in zip(x, 100 * CE["actual_claim_share"]):
        ax.text(xi, 0.25, f"{v:.1f}%", ha="center", fontsize=size, color="#0b0b0b", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" to ", "\nto ").replace("up to", "up\nto").replace("under 1", "under\n1").replace("exactly 1", "exactly\n1") for s in CE["exposure_band"]], fontsize=8.5); ax.set_ylim(0, 11); ax.set_ylabel("policies with a claim (%)")
    ax.set_xlabel("years the policy was in force"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left")


def exposure_freq(ax):
    x = np.arange(len(CE)); w = 0.38
    ax.bar(x - w / 2, CE["actual_frequency"], width=w * 0.92, color=GRAY, label="actual"); ax.bar(x + w / 2, CE["poisson_frequency"], width=w * 0.92, color=ORANGE, label="Poisson model")
    for xi, v in zip(x - w / 2, CE["actual_frequency"]):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" to ", "\nto ").replace("up to", "up\nto").replace("under 1", "under\n1").replace("exactly 1", "exactly\n1") for s in CE["exposure_band"]], fontsize=8.5); ax.set_ylim(0, 0.52); ax.set_ylabel("claims per year in force")
    ax.set_xlabel("years the policy was in force"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")


def importance(ax, size=9):
    t = IM.sort_values("share_of_gain_lost"); y = np.arange(len(t)); ax.barh(y, 100 * t["share_of_gain_lost"], color=[ORANGE if v > 0.2 else BLUE for v in t["share_of_gain_lost"]], height=0.66)
    for yi, v in zip(y, 100 * t["share_of_gain_lost"]):
        ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=size, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(t["variable_group"]); ax.set_xlim(0, 50); ax.set_xlabel("share of the model's out-of-sample gain lost if the variable is dropped (%)"); ax.grid(axis="y", visible=False)


def effects(ax):
    items = [(f"bonus-malus: {l}", f"bonus-malus {l}") for l in ("51-59", "60-79", "80-99", "100+")] + [(f"vehicle age: {l}", f"vehicle age {l}") for l in ("0", "6-10", "11-15", "16+")]; y = np.arange(len(items))[::-1]
    rf = np.array([CF.loc[t, "relative_frequency"] for t, _ in items]); lo = np.array([CF.loc[t, "ci_low"] for t, _ in items]); hi = np.array([CF.loc[t, "ci_high"] for t, _ in items])
    ax.hlines(y, lo, hi, color=INK, alpha=0.6, linewidth=1.5); ax.scatter(rf, y, s=60, color=[ORANGE if v > 1 else BLUE for v in rf], zorder=4)
    for yi, v, h in zip(y, rf, hi):
        ax.text(h * 1.06, yi, f"{v:.2f}×", va="center", fontsize=9, color=INK)
    ax.axvline(1, color=DIM, linestyle="--", linewidth=1); ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5×", "1×", "2×", "4×"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xlim(0.45, 6.5)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in items]); ax.grid(axis="y", visible=False); ax.set_xlabel("claim frequency relative to the reference level (bonus-malus 50, vehicle age 1 to 5), others held fixed; line: 95% interval")


def age(ax):
    m = MG[MG["variable"] == "driver age"].set_index("level"); ref = m.loc["41-50", "frequency"]; raw = [m.loc[l, "frequency"] / ref for l in AGE_LEVELS]; adj = [1.0 if l == "41-50" else CF.loc[f"driver age: {l}", "relative_frequency"] for l in AGE_LEVELS]
    x = np.arange(len(AGE_LEVELS)); w = 0.38; ax.bar(x - w / 2, raw, width=w * 0.92, color=GRAY, label="raw: claims per year relative to ages 41 to 50"); ax.bar(x + w / 2, adj, width=w * 0.92, color=ORANGE, label="adjusted for the other variables, bonus-malus above all")
    for xi, v in zip(x - w / 2, raw):
        ax.text(xi, v + 0.04, f"{v:.2f}×", ha="center", fontsize=8.5, color=INK)
    for xi, v in zip(x + w / 2, adj):
        ax.text(xi, v + 0.04, f"{v:.2f}×", ha="center", fontsize=8.5, color=INK)
    ax.axhline(1, color=DIM, linestyle="--", linewidth=1); ax.set_xticks(x); ax.set_xticklabels(AGE_LEVELS); ax.set_ylim(0, 2.9); ax.set_xlabel("driver age"); ax.set_ylabel("relative claim frequency"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper right")


def calibration(ax):
    ax.plot([0.03, 0.31], [0.03, 0.31], color=DIM, linestyle="--", linewidth=1); ax.scatter(CD["predicted_frequency"], CD["actual_frequency"], s=60, color=ORANGE, zorder=4)
    for _, r in CD.iterrows():
        if r["decile"] in (1, 5, 10):
            ax.text(r["predicted_frequency"] * 1.04, r["actual_frequency"] * 0.93, f"decile {int(r['decile'])}", fontsize=8.5, color=INK, va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.05, 0.1, 0.2]); ax.set_xticklabels(["0.05", "0.1", "0.2"]); ax.set_yticks([0.05, 0.1, 0.2]); ax.set_yticklabels(["0.05", "0.1", "0.2"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("predicted claims per year in force (out of fold)"); ax.set_ylabel("actual claims per year in force"); ax.set_title("Ten equal slices of exposure, ranked by predicted risk")


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 8)); g = fig.add_gridspec(1, 2, width_ratios=[1.1, 1]); a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1])
    exposure_share(a, 10); a.set_title("Share of policies with a claim, by time in force", fontsize=14); importance(b, 10); b.set_title("What the model needs: out-of-sample, one variable dropped", fontsize=14)
    fig.suptitle("What predicts a car insurance claim? Vehicle age and bonus-malus; driver age adds little", x=0.01, ha="left", fontsize=20, color=INK, y=0.98); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0)); exposure_share(axes[0]); axes[0].set_title("Policies with a claim", fontsize=12); exposure_freq(axes[1]); axes[1].set_title("Claims per year in force", fontsize=12); save(fig, "01_exposure.png")
    fig, ax = plt.subplots(figsize=(10, 3.9)); effects(ax); ax.set_title("What the model says: relative claim frequency"); save(fig, "02_effects.png")
    fig, ax = plt.subplots(figsize=(10, 3.8)); age(ax); ax.set_title("Young drivers look riskiest until bonus-malus is held fixed"); save(fig, "03_age.png")
    fig, ax = plt.subplots(figsize=(10, 3.6)); importance(ax); ax.set_title("Which variables the out-of-sample fit needs"); save(fig, "04_importance.png")
    fig, ax = plt.subplots(figsize=(10, 3.9)); calibration(ax); save(fig, "05_calibration.png")


if __name__ == "__main__":
    main()
