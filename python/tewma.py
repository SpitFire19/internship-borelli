import gudhi as gd
import numpy as np
import pandas as pd


def sup_dist(x, y) -> float:
    return np.max(np.abs(x - y))


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


def tewma_detection_vectorized(
    df: pd.DataFrame, gammas: list, window_size=20, T=-1
) -> np.ndarray:

    n_samples = len(df)

    alpha = 0.05
    tau = 2  # Maximum filtration value (a^2 max)
    d = 250  # Discretization

    epsilons = np.linspace(0.1, tau, d)

    phi = np.zeros(d)
    ecc_on_window_prev = []

    gammas_to_check = set(gammas)
    gammas_res = {}

    res = []

    # Initialize stopping time for every threshold
    for gamma in gammas:
        gammas_res.setdefault(gamma, n_samples)

    # Sequential monitoring
    for i in range(window_size, n_samples - window_size + 1):

        # Add a previous non-overlapping window to the reference ECCs
        if i % window_size == 0:

            prev_window = df.iloc[i - window_size : i].to_numpy()

            alpha_st_prev = gd.AlphaComplex(points=prev_window).create_simplex_tree()

            ecc_on_window_prev.append(ecc_on_grid(alpha_st_prev, epsilons))

        # Current sliding window
        window = df.iloc[i : i + window_size].to_numpy()

        alpha_st = gd.AlphaComplex(points=window).create_simplex_tree()

        ecc = ecc_on_grid(alpha_st, epsilons)

        # EWMA update
        phi = alpha * phi + (1 - alpha) * ecc

        # Average reference ECC
        phi_avg = np.mean(ecc_on_window_prev, axis=0)

        # Detection statistic
        distance = sup_dist(phi, phi_avg)

        # Thresholds crossed at this iteration
        to_remove = []

        for gamma in gammas_to_check:

            if distance >= gamma and i > T:
                gammas_res[gamma] = i
                to_remove.append(gamma)

        # Remove thresholds that have already triggered
        for gamma in to_remove:
            gammas_to_check.remove(gamma)

        # Stop early if every threshold has triggered
        if not gammas_to_check:
            break

    # Return results sorted by gamma
    for gamma in sorted(gammas_res.keys()):
        res.append(gammas_res[gamma])

    return np.array(res)
