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
n_epochs = 5
n_samples = 1000
window_size = 20
d = 3
student_df = 5
distributions = [
    rf"$\mathcal{{N}}(0, I_{{{d}}})$",
    r"$MG(0.5)$",
    rf"$\mathcal{{U}}\left([-\sqrt{{3}}, \sqrt{{3}}]^{{{d}}}\right)$",
    rf"$t_{{{student_df}}}^{{{d}}}$",
    rf"$\mathrm{{Laplace}}(0, 1)^{{{d}}}$",
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
i = 0
j = 1

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
n_thresholds = 1000
gammas = np.logspace(0, 100, n_thresholds)
gammas_mmd = np.linspace(0, 5, n_thresholds)
gammas_ed = np.linspace(0, 0.7, n_thresholds)
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

    no_change = generate_dist(rng, null_dist_name, d, n_samples)
    df_no_change = pd.DataFrame(no_change)
    before = generate_dist(rng, null_dist_name, d, n_samples // 2)
    after = generate_dist(rng, alt_dist_name, d, n_samples // 2)
    changed = np.concatenate(
        [before, after],
        axis=0,
    )
    df_change = pd.DataFrame(changed)

    # ========================================================
    # Run selected algorithm
    # ========================================================
    arl = tewma_detection_vectorized(
        df=df_no_change,
        gammas=gammas,
        window_size=window_size,
    )
    dd = tewma_detection_vectorized(
        df=df_change,
        gammas=gammas,
        window_size=window_size,
        T=n_samples // 2,
    )
    dd = dd - n_samples // 2
    arl_total["TEWMA"] += arl
    dd_total["TEWMA"] += dd
    # -------------------------
    # MMD
    # -------------------------
    arl = mmd_detection_vectorized(
        df=df_no_change,
        gammas=gammas_mmd,
        window_size=window_size,
    )
    dd = mmd_detection_vectorized(
        df=df_change,
        gammas=gammas_mmd,
        window_size=window_size,
        T=n_samples // 2,
    )
    dd = dd - n_samples // 2
    arl_total["MMD"] += arl
    dd_total["MMD"] += dd
    # -------------------------
    # Energy distance
    # -------------------------
    arl = ed_detection_vectorized(
        X=df_no_change,
        thresholds=gammas_ed,
        window_size=window_size,
    )
    dd = ed_detection_vectorized(
        X=df_change,
        thresholds=gammas_ed,
        window_size=window_size,
        T=n_samples // 2,
    )
    print(arl.shape)
    print(arl[:10])
    print(arl[-10:])
    print(np.isinf(arl).sum())
    dd = dd - n_samples // 2
    arl_total["Energy-based"] += arl
    dd_total["Energy-based"] += dd


for algo in algorithms:
    arl_total[algo] /= n_epochs
    dd_total[algo] /= n_epochs
    print(arl_total[algo].max())
    print(dd_total[algo].max())
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
