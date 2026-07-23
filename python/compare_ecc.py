import os
from concurrent.futures import ProcessPoolExecutor

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from changepoint.ed import ed_detection_vectorized
from changepoint.mmd import mmd_detection_vectorized
from changepoint.tewma import tewma_detection_vectorized
from generate_data import generate_dist


def ecc_on_grid(st, filtration_grid):
    simplices = list(st.get_filtration())
    idx = 0
    chi = 0
    curve = []

    for eps in filtration_grid:
        while idx < len(simplices) and simplices[idx][1] <= eps:
            simplex, _ = simplices[idx]
            chi += (-1) ** (len(simplex) - 1)
            idx += 1
        curve.append(chi)

    return np.array(curve)


def get_all_filtrations(st):
    filtrations = []
    for simplex, filtration in st.get_filtration():
        filtrations.append(filtration)
    return np.array(sorted(filtrations))


def cut_filtration_to_fit(filtration: list, ecc: list) -> list:
    last_value = 1
    for idx in range(len(ecc) - 1, -1, -1):
        if ecc[idx] != last_value:
            return idx + 1
    return len(ecc)


def cut_both_to_fit(n_points: int, filtration: np.array, ecc: np.array):
    last_idx = cut_filtration_to_fit(filtration, ecc)
    last_val = filtration[last_idx]
    return filtration[:last_idx] / last_val, ecc[:last_idx] / n_points


# ============================================================
# Configuration
# ============================================================

n_samples = 2000
window_size = 200
d = 3
student_df = 5
rng = np.random.default_rng(42)

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
        null_dist = distributions[i]
        alt_dist = distributions[j]
        null_dist_name = distribution_names[i]
        alt_dist_name = distribution_names[j]

        null_dist_window = generate_dist(rng, null_dist_name, d, window_size)
        alt_dist_window = generate_dist(rng, alt_dist_name, d, window_size)

        null_dist_st = gd.DelaunayCechComplex(
            points=null_dist_window
        ).create_simplex_tree()
        filtration = get_all_filtrations(null_dist_st)
        ecc_null_dist = ecc_on_grid(null_dist_st, filtration)
        filt_null, ecc_null = cut_both_to_fit(window_size, filtration, ecc_null_dist)
        ax.plot(filt_null, ecc_null)

        alt_dist_st = gd.DelaunayCechComplex(
            points=alt_dist_window
        ).create_simplex_tree()
        filtration = get_all_filtrations(alt_dist_st)
        ecc_alt_dist = ecc_on_grid(alt_dist_st, filtration)
        filt_alt, ecc_alt = cut_both_to_fit(window_size, filtration, ecc_alt_dist)
        ax.plot(filt_alt, ecc_alt)


fsize = 30

# ============================================================
# Distribution names
# ============================================================
for j, dist in enumerate(distributions[:n]):
    axes[-1, j].set_xlabel(
        dist,
        fontsize=12,
        rotation=0,
        labelpad=10,
    )

for i, dist in enumerate(distributions[:n]):
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

plt.subplots_adjust(
    left=0.12,
    right=0.98,
    bottom=0.12,
    top=0.92,
    wspace=0.15,
    hspace=0.15,
)
plt.show()
