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
# Distribution names used by generate_dist
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

n_epochs = 50
n = len(distributions)


# ============================================================
# Random number generator
# ============================================================

rng = np.random.default_rng(42)


# ============================================================
# DEBUGGING CONFIGURATION
#
# Select exactly one pair (i, j) and one algorithm
# ============================================================

i = 0
j = 1

selected_algorithm = "TEWMA"

# Examples:
#
# i = 0, j = 1:
# N(0, I_d) -> MG(0.5)
#
# i = 2, j = 4:
# Uniform -> Laplace
#
# Possible algorithms:
# "TEWMA"
# "MMD"
# "Energy-based"


# ============================================================
# Check selected pair
# ============================================================

if i == j:
    raise ValueError(
        "For debugging, choose i != j because diagonal pairs "
        "correspond to no distributional change."
    )

if selected_algorithm not in algorithms:
    raise ValueError(
        f"Unknown algorithm: {selected_algorithm!r}. " f"Choose one of {algorithms}."
    )


# ============================================================
# Get selected null and alternative distributions
# ============================================================

null_dist = distributions[i]
alt_dist = distributions[j]

null_dist_name = distribution_names[i]
alt_dist_name = distribution_names[j]


print("=" * 60)
print("DEBUGGING CONFIGURATION")
print("=" * 60)
print(f"Pair indices:         ({i}, {j})")
print(f"Null distribution:    {null_dist_name}")
print(f"Alternative dist.:    {alt_dist_name}")
print(f"Algorithm:            {selected_algorithm}")
print(f"Dimension:            {d}")
print(f"Samples per dist.:    {n_samples}")
print(f"Window size:          {window_size}")
print(f"Number of epochs:     {n_epochs}")
print("=" * 60)


# ============================================================
# Threshold configuration
# ============================================================

n_values = 1000

gammas = np.linspace(0, 20, n_values)

# ============================================================
# Initialize totals BEFORE the epoch loop
# ============================================================

arl_total = np.zeros(n_values, dtype=float)
dd_total = np.zeros(n_values, dtype=float)

# ============================================================
# Run Monte Carlo epochs
# ============================================================
rng = np.random.default_rng(42)
for epoch in range(n_epochs):

    print(f"\nEpoch {epoch + 1}/{n_epochs}")

    # --------------------------------------------------------
    # Generate pre-change data
    # --------------------------------------------------------

    null_dist_data = generate_dist(rng, null_dist_name, d, n_samples // 2)

    # --------------------------------------------------------
    # Generate post-change data
    # --------------------------------------------------------

    alt_dist_data = generate_dist(rng, alt_dist_name, d, n_samples // 2)

    # --------------------------------------------------------
    # Concatenate:
    #
    # rows 0, ..., n_samples - 1:
    #     null distribution
    #
    # rows n_samples, ..., 2*n_samples - 1:
    #     alternative distribution
    # --------------------------------------------------------

    data = np.concatenate([null_dist_data, alt_dist_data], axis=0)

    data_df = pd.DataFrame(data)

    print(f"Data shape: {data_df.shape}")

    # ========================================================
    # Run selected algorithm
    # ========================================================

    if selected_algorithm == "TEWMA":
        arl = tewma_detection_vectorized(data_df, gammas, window_size)
        dd = tewma_detection_vectorized(data_df, gammas, window_size, T=n_samples // 2)
        dd -= n_samples // 2

    elif selected_algorithm == "MMD":
        arl = mmd_detection_vectorized(data_df, gammas, window_size)
        dd = mmd_detection_vectorized(data_df, gammas, window_size, T=n_samples // 2)
        dd -= n_samples // 2

    elif selected_algorithm == "Energy-based":
        arl = ed_detection_vectorized(data_df, gammas, window_size)
        dd = ed_detection_vectorized(data_df, gammas, window_size, T=n_samples // 2)
        dd -= n_samples // 2
    # ========================================================
    # Convert detector output to NumPy arrays
    # ========================================================
    #
    # If your detector returns a dictionary:
    #
    #     {gamma_1: stopping_time_1, ...}
    #
    # then extract values in exactly the same order as gammas.
    # ========================================================

    if isinstance(arl, dict):
        arl = np.array([arl[gamma] for gamma in gammas], dtype=float)
    else:
        arl = np.asarray(arl, dtype=float)

    if isinstance(dd, dict):
        dd = np.array([dd[gamma] for gamma in gammas], dtype=float)
    else:
        dd = np.asarray(dd, dtype=float)

    # ========================================================
    # Debug information
    # ========================================================

    print(f"ARL shape: {arl.shape}")
    print(f"DD shape:  {dd.shape}")

    print("First 5 ARL values:", arl[:5])

    print("First 5 DD values:", dd[:5])

    # ========================================================
    # Accumulate results
    # ========================================================
    arl_total += arl
    dd_total += dd


# ============================================================
# Average across epochs
# ============================================================

arl_avg = arl_total / n_epochs
dd_avg = dd_total / n_epochs


# ============================================================
# Print final debugging results
# ============================================================

print("\n" + "=" * 60)
print("FINAL AVERAGED RESULTS")
print("=" * 60)

print("ARL average:")
print(arl_avg)

print("\nDD average:")
print(dd_avg)


# ============================================================
# Create ONE plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))


# ============================================================
# Plot selected algorithm
# ============================================================

ax.plot(
    arl_avg, dd_avg, marker="o", markersize=4, linewidth=1.5, label=selected_algorithm
)


# ============================================================
# Axis labels
# ============================================================

ax.set_xlabel("Average Run Length (ARL)", fontsize=13)

ax.set_ylabel("Detection Delay (DD)", fontsize=13)


# ============================================================
# Title
# ============================================================

ax.set_title(
    f"{selected_algorithm}: " f"{null_dist} $\\rightarrow$ {alt_dist}", fontsize=14
)


# ============================================================
# Make sure numerical tick labels are visible
# ============================================================

ax.tick_params(
    axis="both", which="major", labelsize=11, labelbottom=True, labelleft=True
)


# ============================================================
# Grid and legend
# ============================================================

ax.grid(True, alpha=0.3)

ax.legend(fontsize=11)


# ============================================================
# Adjust layout and show
# ============================================================

plt.tight_layout()
plt.show()
