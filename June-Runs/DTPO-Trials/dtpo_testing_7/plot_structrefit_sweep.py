"""
Single aggregate plot of survival curves for bus5 DTPO structure-refit interval sweep (K=1,3,5,10).
"""
import re
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/home/sai/RL2Grid_full/dtpo_testing_7"
KS = [1, 3, 5, 10]
SEEDS = [100, 101, 102, 103]

COLORS = {1: "#2a78d6", 3: "#1baf7a", 5: "#eda100", 10: "#008300"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

LINE_RE = re.compile(r"it=(\d+)/(\d+).*?survival=([\d.]+)%")


def load_curve(k, seed):
    path = f"{ROOT}/bus5_structrefit_K{k}_seed{seed}/train.log"
    its, survs = [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                its.append(int(m.group(1)))
                survs.append(float(m.group(3)))
    return np.array(its), np.array(survs)


fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for k in KS:
    color = COLORS[k]
    all_curves = []
    for seed in SEEDS:
        its, survs = load_curve(k, seed)
        if len(its) == 0:
            continue
        ax.plot(its, survs, color=color, linewidth=0.8, alpha=0.22, zorder=2)
        all_curves.append((its, survs))

    max_len = max(len(s) for _, s in all_curves)
    stacked = np.full((len(all_curves), max_len), np.nan)
    for i, (its, survs) in enumerate(all_curves):
        stacked[i, : len(survs)] = survs
    mean_curve = np.nanmean(stacked, axis=0)
    x_mean = np.arange(1, max_len + 1)
    ax.plot(x_mean, mean_curve, color=color, linewidth=2.8, alpha=1.0,
             zorder=3, label=f"K={k} (mean of {len(all_curves)} seeds)")

ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(BASELINE)

ax.set_xlim(1, 150)
ax.set_ylim(0, 103)
ax.set_xlabel("DTPO iteration", color=INK_SECONDARY, fontsize=11)
ax.set_ylabel("Survival (%)", color=INK_SECONDARY, fontsize=11)
ax.tick_params(colors=INK_MUTED, labelsize=9)
ax.set_title(
    "bus5 DTPO: survival per iteration by structure-refit interval K\n"
    "(faint lines = individual seeds 100-103, bold = mean across seeds)",
    color=INK_PRIMARY, fontsize=12.5, pad=50,
)

legend = ax.legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
    frameon=False, fontsize=10, labelcolor=INK_PRIMARY, handlelength=2.2,
    columnspacing=1.4,
)

fig.tight_layout()
out_path = f"{ROOT}/structrefit_sweep_survival_curves.png"
fig.savefig(out_path, facecolor=SURFACE)
print(f"saved: {out_path}")
