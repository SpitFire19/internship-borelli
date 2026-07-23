import argparse
import time
from pathlib import Path

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_all_filtrations(st):
    filtrations = []
    for simplex, filtration in st.get_filtration():
        filtrations.append(filtration)
    return np.array(sorted(filtrations))


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


n_points = 1000
n_dim = 3
df = np.random.randn(n_points, n_dim)

delaunay_cech_st = gd.DelaunayCechComplex(points=df).create_simplex_tree()
filtration = get_all_filtrations(delaunay_cech_st)
ecc_delaunay_cech = ecc_on_grid(delaunay_cech_st, filtration)
# last_idx = cut_filtration_to_fit(filtration, ecc_delaunay_cech)
# print(len(filtration), last_idx)
# last_val = filtration[last_idx]
# plt.plot(filtration[:last_idx] / last_val, ecc_delaunay_cech[:last_idx] / n_points)
a, b = cut_both_to_fit(n_points, filtration, ecc_delaunay_cech)
plt.plot(a, b + 1)
fsize = 30
plt.title(
    f"Euler curve for time series of length n = 1000",
    fontsize=fsize,
)
plt.xlabel("Filtration value(normalized) r", fontsize=fsize)
plt.ylabel("ECC(r)", fontsize=fsize)
plt.tight_layout()

plt.show()
