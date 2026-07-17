import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    # Average pairwise distance between X and Y
    cross_distance = cdist(X, Y, metric="euclidean").mean()
    # Average pairwise distance within X
    within_X = cdist(X, X, metric="euclidean").mean()
    # Average pairwise distance within Y
    within_Y = cdist(Y, Y, metric="euclidean").mean()
    return 2.0 * cross_distance - within_X - within_Y


def ed_detection_vectorized(
    X: pd.DataFrame,
    window_size: int,
    thresholds: np.ndarray,
    T: int = -1,
) -> np.ndarray:

    n_samples = len(X)
    thresholds = np.asarray(
        thresholds,
        dtype=float,
    ).ravel()

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("All thresholds must be finite.")
    X_values = X.to_numpy(dtype=float)
    if not np.all(np.isfinite(X_values)):
        raise ValueError("X contains NaN or infinite values.")

    # ========================================================
    # Initialization
    # ========================================================

    w = window_size
    gammas_res = {gamma: np.inf for gamma in thresholds}
    # ========================================================
    # Sequential detection
    # ========================================================
    for t in range(2 * w, n_samples + 1):
        reference_window = X_values[t - 2 * w : t - w]
        test_window = X_values[t - w : t]
        statistic = energy_distance(
            reference_window,
            test_window,
        )
        if t <= T:
            continue
        for gamma in gammas_res:
            if gammas_res[gamma] == np.inf and statistic > gamma:
                gammas_res[gamma] = t

    # ========================================================
    # Collect stopping times
    # ========================================================
    res = []
    for gamma in sorted(gammas_res.keys()):
        if gammas_res[gamma]:
            res.append(min(gammas_res[gamma], n_samples))

    return np.asarray(res)
