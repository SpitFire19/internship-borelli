import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ed import ed_detection_vectorized
from generate_data import generate_dist
from mmd import mmd_detection_vectorized
from tewma import tewma_detection_vectorized

# ============================================================
# Configuration
# ============================================================
n_epochs = 1
n_samples = 1000
window_size = 50
d = 2
student_df = 5
distributions = [
    rf"$\mathcal{{N}}(0, I_{{{d}}})$",
    r"$MG(0.5)$",
    rf"$\mathcal{{U}}\left([-\sqrt{{3}}, \sqrt{{3}}]^{{{d}}}\right)$",
    rf"$t_{{{student_df}}}^{{{d}}}$",
    rf"$\mathrm{{Laplace}}(0,1)^{{{d}}}$",
    rf"$\mathrm{{Dir}}(1, 1, 1)$",
    rf"$\mathrm{{Dir}}(5, 5, 5)$",
]
# ============================================================
# Distribution names used by generate_dist file
# ============================================================

distribution_names = [
    "N(0, 1)",
    "MG(0.5)",
    "U[-sqrt(3), sqrt(3)]",
    "t(5)",
    "Laplace(0, 1)",
    "Dir(1, 1, 1)",
    "Dir(10, 10, 10)",
]

# ============================================================
# Change-point detection algorithms
# ============================================================
algorithms_count = 3
algorithms = [
    "TEWMA",
    "MMD",
    "Energy-based",
]
fig, ax = plt.subplots(figsize=(8, 6))

markers = {
    "TEWMA": "o",
    "MMD": "s",
    "Energy-based": "^",
}


n = len(distributions)
rng = np.random.default_rng(42)
# ============================================================
# DEBUGGING CONFIGURATION
# Select exactly one pair (i, j) and one algorithm
# ============================================================
i = 5
j = 6

# ============================================================
# Get selected null and alternative distributions
# ============================================================
null_dist = distributions[i]
alt_dist = distributions[j]
null_dist_name = distribution_names[i]
alt_dist_name = distribution_names[j]

# ============================================================
# Threshold configuration
# ============================================================
n_thresholds = 100
gammas_tewma = np.linspace(0, 0.08, n_thresholds)
gammas_mmd = np.linspace(0, 5, n_thresholds)
gammas_ed = np.linspace(0, 0.8, n_thresholds)
# ============================================================
# Initialize totals BEFORE the epoch loop
# ============================================================
arl_total = {algo: np.zeros(n_thresholds) for algo in algorithms}
dd_total = {algo: np.zeros(n_thresholds) for algo in algorithms}

# ============================================================
# Run Monte Carlo epochs
# ============================================================
rng = np.random.default_rng(42)
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")

    before = generate_dist(rng, null_dist_name, d, n_samples // 2)
    no_change_after = generate_dist(rng, null_dist_name, d, n_samples // 2)
    change_after = generate_dist(rng, alt_dist_name, d, n_samples // 2)
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

    # Run selected algorithm
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
    print(arl_tewma.shape)
    print(*arl_tewma)
    dd_ed -= n_samples // 2
    arl_total["Energy-based"] += arl_ed
    dd_total["Energy-based"] += dd_ed


for algo in algorithms:
    arl_total[algo] /= n_epochs
    dd_total[algo] /= n_epochs
    ax.plot(
        arl_total[algo],
        dd_total[algo],
        linewidth=1.5,
        marker=markers[algo],
        markersize=4,
        label=algo,
    )

ax.set_xlabel("Average Run Length (ARL)", fontsize=13)
ax.set_ylabel("Detection Delay (DD)", fontsize=13)

ax.set_title(
    f"{null_dist} → {alt_dist}",
    fontsize=15,
)

ax.grid(True, alpha=0.3)

ax.legend(
    title="Algorithm",
    fontsize=11,
    title_fontsize=12,
    loc="best",
)

plt.tight_layout()
plt.show()
