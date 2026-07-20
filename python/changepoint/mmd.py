import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


def compute_mmd(X, Y, gamma):
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    return np.sqrt(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def anomaly_detection_mmd(
    df: pd.DataFrame,
    gamma: float,
    window_size=20,
) -> int:

    n_samples = len(df)
    mmd_prev = []

    for i in range(window_size, n_samples - window_size + 1):
        current_window = df.iloc[i : i + window_size].drop(columns="Date").to_numpy()
        prev_window = df.iloc[i - window_size : i].drop(columns="Date").to_numpy()
        gamma_kernel = 0.01
        mmd = compute_mmd(prev_window, current_window, gamma_kernel)
        if i % window_size == 1 and i > 1:
            mmd_prev.append(mmd)
        if len(mmd_prev) == 0:
            continue
        mmd_avg = np.mean(mmd_prev)
        if abs(mmd) >= gamma:
            return i

    return n_samples


def mmd_detection_vectorized(
    df: pd.DataFrame, gammas: list, window_size=20, T=-1
) -> np.ndarray:

    n_samples = len(df)
    mmd_prev = []
    gammas_to_check = set(gammas)
    gammas_res = {}
    res = []
    for gamma in gammas:
        gammas_res.setdefault(gamma, n_samples)
    for i in range(window_size, n_samples - window_size + 1):
        current_window = df.iloc[i : i + window_size].to_numpy()
        prev_window = df.iloc[i - window_size : i].to_numpy()
        gamma_kernel = 0.001
        mmd = compute_mmd(prev_window, current_window, gamma_kernel)
        if i % window_size == 1 and i > 1:
            mmd_prev.append(mmd)
        if len(mmd_prev) == 0:
            continue
        to_remove = []
        distance = abs(mmd)
        for gamma in gammas_to_check:
            if distance >= gamma and i > T:
                gammas_res[gamma] = i
                to_remove.append(gamma)
        for gamma in to_remove:
            gammas_to_check.remove(gamma)
        if not gammas_to_check:
            break
    for gamma in sorted(gammas_res.keys()):
        res.append(gammas_res[gamma])
    return np.array(res)
