import matplotlib.pyplot as plt
import numpy as np
from generate_data import generate_dist

# Settings
rng = np.random.default_rng(42)

null_dist = "Dir(10, 10, 10)"
alt_dist = "Dir(1, 1, 1)"

dim = 2
n_samples = 2000

# -----------------------------
# Generate data
# -----------------------------
X0 = generate_dist(
    rng,
    null_dist,
    dim + 1,
    n_samples,
)

X1 = generate_dist(
    rng,
    alt_dist,
    dim + 1,
    n_samples,
)

# -----------------------------
# Visualize
# -----------------------------
if dim == 1:

    plt.figure(figsize=(8, 5))

    plt.hist(
        X0[:, 0],
        bins=40,
        density=True,
        alpha=0.5,
        label=null_dist,
    )

    plt.hist(
        X1[:, 0],
        bins=40,
        density=True,
        alpha=0.5,
        label=alt_dist,
    )

    plt.xlabel("x")
    plt.ylabel("Density")

elif dim == 2:

    plt.figure(figsize=(7, 7))

    plt.scatter(
        X0[:, 0],
        X0[:, 1],
        s=8,
        alpha=0.5,
        label=null_dist,
    )
    """
    plt.scatter(
        X1[:, 0],
        X1[:, 1],
        s=8,
        alpha=0.5,
        label=alt_dist,
    )
    """
    plt.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-", lw=2)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

elif dim == 3:

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X0[:, 0],
        X0[:, 1],
        X0[:, 2],
        s=6,
        alpha=0.4,
        label=null_dist,
    )

    ax.scatter(
        X1[:, 0],
        X1[:, 1],
        X1[:, 2],
        s=6,
        alpha=0.4,
        label=alt_dist,
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")

else:

    plt.figure(figsize=(7, 7))

    plt.scatter(
        X0[:, 0],
        X0[:, 1],
        s=6,
        alpha=0.5,
        label=null_dist,
    )

    plt.scatter(
        X1[:, 0],
        X1[:, 1],
        s=6,
        alpha=0.5,
        label=alt_dist,
    )

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(f"First two coordinates (dim={dim})")

plt.title(f"{null_dist}")
plt.legend()
plt.tight_layout()
plt.show()
