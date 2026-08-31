"""
Three figures for the gate+replay (dtpo_testing_8) vs baseline (dtpo_testing_7) comparison

1. gatereplay_sweep_survival_curves.png — dtpo_testing_8 bus5 K-sweep aggregate (deployed survival vs iteration, colored by K, faint seeds + bold mean).
2. before_after_bus5.png — side-by-side: dtpo_testing_7 raw curves (before) vs dtpo_testing_8 deployed curves (after), same axes/colors.
3. before_after_bus14.png — old bus14 16-leaf run vs the three new gated seeds.
"""
import re
import numpy as np
import matplotlib.pyplot as plt

OLD = "/home/sai/RL2Grid_full/dtpo_testing_7"
NEW = "/home/sai/RL2Grid_full/dtpo_testing_8"
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


def load_curve(path):
    its, survs = [], []
    try:
        with open(path) as f:
            for line in f:
                m = LINE_RE.search(line)
                if m:
                    its.append(int(m.group(1)))
                    survs.append(float(m.group(3)))
    except FileNotFoundError:
        pass
    return np.array(its), np.array(survs)


def style_axis(ax, title):
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
    ax.set_title(title, color=INK_PRIMARY, fontsize=11.5)


def draw_k_sweep(ax, root, pattern):
    for k in KS:
        color = COLORS[k]
        all_curves = []
        for seed in SEEDS:
            its, survs = load_curve(f"{root}/{pattern.format(k=k, seed=seed)}/train.log")
            if len(its) == 0:
                continue
            ax.plot(its, survs, color=color, linewidth=0.8, alpha=0.22, zorder=2)
            all_curves.append(survs)
        if not all_curves:
            continue
        max_len = max(len(s) for s in all_curves)
        stacked = np.full((len(all_curves), max_len), np.nan)
        for i, s in enumerate(all_curves):
            stacked[i, : len(s)] = s
        ax.plot(np.arange(1, max_len + 1), np.nanmean(stacked, axis=0), color=color,
                linewidth=2.6, zorder=3, label=f"K={k}")


# Figure 1: dtpo_testing_8 aggregate K-sweep
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
draw_k_sweep(ax, NEW, "bus5_gatereplay_K{k}_seed{seed}")
style_axis(ax, "bus5 DTPO with survival gate + replay: deployed survival per iteration by K\n"
               "(faint = seeds 100-103, bold = mean per K; curves are monotone by construction)")
ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK_PRIMARY)
fig.tight_layout()
fig.savefig(f"{NEW}/gatereplay_sweep_survival_curves.png", facecolor=SURFACE)
print("saved figure 1")

# Figure 2: before/after bus5, side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=150, sharey=True)
fig.patch.set_facecolor(SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
draw_k_sweep(axes[0], OLD, "bus5_structrefit_K{k}_seed{seed}")
draw_k_sweep(axes[1], NEW, "bus5_gatereplay_K{k}_seed{seed}")
style_axis(axes[0], "BEFORE (dtpo_testing_7): surrogate-only acceptance\n"
                    "live-policy survival — spikes and collapses")
style_axis(axes[1], "AFTER (dtpo_testing_8): survival gate + replay\n"
                    "deployed-policy survival — monotone staircases")
axes[1].set_ylabel("")
axes[0].legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK_PRIMARY, ncol=2)
fig.suptitle("bus5 DTPO training curves, 16 runs each (K x seed), 16-leaf trees",
             color=INK_PRIMARY, fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(f"{NEW}/before_after_bus5.png", facecolor=SURFACE)
print("saved figure 2")

# Figure 3: before/after bus14
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
its, survs = load_curve(f"{OLD}/bus14_t0.5_d0_seed100/train.log")
ax.plot(its, survs, color="#c33d35", linewidth=2.0, linestyle="--", zorder=2,
        label="before: seed100, no gate (eval every 10 it; honest 63.6%)")
its, survs = load_curve(f"{OLD}/bus14_t0.5_d0_seed100_leaves64_evalevery1/train.log")
if len(its):
    ax.plot(its, survs, color="#c33d35", linewidth=0.9, alpha=0.35, zorder=2,
            label="before: seed100, 64 leaves, no gate (per-it eval; collapses)")
new_labels = {100: "63.6%", 101: "59.2%", 102: "100.0%"}
for seed, color in zip([100, 101, 102], ["#2a78d6", "#1baf7a", "#eda100"]):
    its, survs = load_curve(f"{NEW}/bus14_gatereplay_K5_seed{seed}/train.log")
    if len(its):
        ax.plot(its, survs, color=color, linewidth=2.2, zorder=3,
                label=f"after: seed{seed}, gate+replay K=5 (honest80 {new_labels[seed]})")
style_axis(ax, "bus14 DTPO, 16-leaf trees: before (no gate) vs after (survival gate + replay)\n"
               "deployed survival on the in-loop eval set (before: 20 chronics; after: 40)")
ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_PRIMARY)
fig.tight_layout()
fig.savefig(f"{NEW}/before_after_bus14.png", facecolor=SURFACE)
print("saved figure 3")
