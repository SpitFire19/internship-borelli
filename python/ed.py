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
    X: pd.DataFrame, window_size: int, thresholds: np.ndarray, T=-1
) -> np.ndarray:
    # ========================================================
    # Input validation
    # ========================================================
    if len(X) < 2 * window_size:
        raise ValueError(
            f"X contains {len(X)} observations, but at least "
            f"{2 * window_size} are required."
        )

    # Convert thresholds to a 1D NumPy array
    thresholds = np.asarray(thresholds, dtype=float).ravel()

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("All thresholds must be finite.")

    # Convert DataFrame to NumPy once, outside the loop
    X_values = X.to_numpy(dtype=float)

    if not np.all(np.isfinite(X_values)):
        raise ValueError("X contains NaN or infinite values.")

    # ========================================================
    # Initialization
    # ========================================================
    n_samples = len(X_values)
    w = window_size
    n_thresholds = len(thresholds)
    # np.inf means that no alarm has occurred yet
    stopping_times = np.full(n_thresholds, np.inf)
    # True while a threshold has not yet been crossed
    active = np.ones(n_thresholds, dtype=bool)

    # ========================================================
    # Sequential detection
    # ========================================================

    for t in range(2 * w, n_samples + 1):
        # Previous/reference window
        reference_window = X_values[t - 2 * w : t - w]
        # Current/test window
        test_window = X_values[t - w : t]
        # Compute the energy statistic ONCE for this time step
        statistic = energy_distance(reference_window, test_window)
        if t <= T:
            continue
        # Compare against all currently active thresholds
        newly_crossed = active & (statistic > thresholds)
        # Store t only for thresholds crossed for the first time
        stopping_times[newly_crossed] = t
        # Deactivate thresholds that have been crossed
        active[newly_crossed] = False

        # If every threshold has been crossed, stop early
        if not np.any(active):
            break

    return stopping_times
