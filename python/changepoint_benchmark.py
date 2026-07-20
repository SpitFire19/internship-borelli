import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from changepoint.ed import ed_detection_vectorized
from changepoint.mmd import mmd_detection_vectorized
from changepoint.tewma import tewma_detection_vectorized
from generate_data import generate_dist

# ============================================================
# Configuration
# ============================================================

n_samples = 1000
window_size = 50
d = 2
student_df = 5

n_epochs = 25
n_thresholds = 200

rng = np.random.default_rng(42)

algorithms = ["TEWMA", "MMD", "Energy-based"]

distributions = [
    rf"$\mathcal{{N}}(0, I_{{{d}}})$",
    r"$MG(0.5)$",
    rf"$\mathcal{{U}}\left([-\sqrt{{3}}, \sqrt{{3}}]^{{{d}}}\right)$",
    rf"$t_{{{student_df}}}^{{{d}}}$",
    rf"$\mathrm{{Laplace}}(0,1)^{{{d}}}$",
    rf"$\mathrm{{Dir}}(1, 1, 1)$",
    rf"$\mathrm{{Dir}}(10, 10, 10)$",
]
distribution_names = [
    "N(0, 1)",
    "MG(0.5)",
    "U[-sqrt(3), sqrt(3)]",
    "t(5)",
    "Laplace(0, 1)",
    "Dir(1, 1, 1)",
    "Dir(10, 10, 10)",
]
n = len(distributions)

# ============================================================
# Threshold configuration
# ============================================================
n_thresholds = 1000
gammas_tewma = np.logspace(0, 100, n_thresholds)
gammas_mmd = np.linspace(0, 5, n_thresholds)
gammas_ed = np.linspace(0, 0.8, n_thresholds)

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    n,
    n,
    figsize=(18, 16),
    sharex=True,
    sharey=True,
)

# ============================================================
# Main loop
# ============================================================

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == j:
            ax.plot(
                [0, 1],
                [0, 1],
                transform=ax.transAxes,
                color="black",
                lw=1.5,
            )
            ax.plot(
                [0, 1],
                [1, 0],
                transform=ax.transAxes,
                color="black",
                lw=1.5,
            )
            ax.grid(False)
            continue
        print(f"Build window ({i + 1}, {j + 1})")
        null_name = distribution_names[i]
        alt_name = distribution_names[j]

        arl_total = {algo: np.zeros(n_thresholds) for algo in algorithms}
        dd_total = {algo: np.zeros(n_thresholds) for algo in algorithms}

        for epoch in range(n_epochs):
            before = generate_dist(rng, null_name, d, n_samples // 2)
            no_change_after = generate_dist(rng, null_name, d, n_samples // 2)
            change_after = generate_dist(rng, alt_name, d, n_samples // 2)
            no_change = np.concatenate(
                [before, no_change_after],
                axis=0,
            )
            changed = np.concatenate(
                [before, change_after],
                axis=0,
            )
            df_no_change = pd.DataFrame(no_change)
            df_change = pd.DataFrame(changed)

            # TEWMA
            arl_tewma = tewma_detection_vectorized(
                df=df_no_change,
                gammas=gammas_tewma,
                window_size=window_size,
            )
            dd_tewma = tewma_detection_vectorized(
                df=df_change,
                gammas=gammas_tewma,
                window_size=window_size,
                T=n_samples // 2,
            )

            dd_tewma -= n_samples // 2
            arl_total["TEWMA"] += arl_tewma
            dd_total["TEWMA"] += dd_tewma

            # MMD
            arl_mmd = mmd_detection_vectorized(
                df=df_no_change,
                gammas=gammas_mmd,
                window_size=window_size,
            )
            dd_mmd = mmd_detection_vectorized(
                df=df_change,
                gammas=gammas_mmd,
                window_size=window_size,
                T=n_samples // 2,
            )

            dd_mmd -= n_samples // 2
            arl_total["MMD"] += arl_mmd
            dd_total["MMD"] += dd_mmd

            # Energy distance
            arl_ed = ed_detection_vectorized(
                X=df_no_change,
                thresholds=gammas_ed,
                window_size=window_size,
            )
            dd_ed = ed_detection_vectorized(
                X=df_change,
                thresholds=gammas_ed,
                window_size=window_size,
                T=n_samples // 2,
            )

            dd_ed -= n_samples // 2
            arl_total["Energy-based"] += arl_ed
            dd_total["Energy-based"] += dd_ed

        # Average over epochs

        for algo in algorithms:
            arl_total[algo] /= n_epochs
            dd_total[algo] /= n_epochs

            ax.plot(
                arl_total[algo],
                dd_total[algo],
                linewidth=1.5,
                marker="o",
                markersize=2,
                label=algo,
            )

        ax.grid(True, alpha=0.3)
        print(f"Finished ({i+1}, {j+1})")

# ============================================================
# Distribution names
# ============================================================
for j, dist in enumerate(distributions):
    axes[-1, j].set_xlabel(
        dist,
        fontsize=12,
        rotation=0,
        labelpad=10,
    )

for i, dist in enumerate(distributions):
    axes[i, 0].set_ylabel(
        dist,
        fontsize=12,
        rotation=0,
        labelpad=30,
        va="center",
    )

# ============================================================
# Tick labels
# ============================================================
for i in range(n):
    axes[i, 0].tick_params(
        axis="y",
        labelleft=True,
    )

for j in range(n):
    axes[-1, j].tick_params(
        axis="x",
        labelbottom=True,
    )

# ============================================================
# Global labels
# ============================================================
fig.text(
    0.5,
    0.04,
    "Alternative distribution",
    ha="center",
    fontsize=16,
)

fig.text(
    0.04,
    0.5,
    "Null distribution",
    rotation="vertical",
    va="center",
    fontsize=16,
)

# ============================================================
# Legend
# ============================================================
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    fontsize=12,
)
fig.suptitle(
    "ARL/DD comparison of change-point detection algorithms",
    fontsize=18,
)
plt.subplots_adjust(
    left=0.12,
    right=0.98,
    bottom=0.12,
    top=0.92,
    wspace=0.15,
    hspace=0.15,
)
plt.show()
