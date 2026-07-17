import numpy as np


def generate_dist(
    rng: np.random.Generator, dist_name: str, dim: int, n_samples: int
) -> np.ndarray:

    if dist_name == "N(0, 1)":
        return rng.normal(loc=0.0, scale=1.0, size=(n_samples, dim))

    elif dist_name == "MG(0.5)":
        rho = 0.5

        covariance = np.full((dim, dim), rho, dtype=float)
        np.fill_diagonal(covariance, 1.0)

        return rng.multivariate_normal(
            mean=np.zeros(dim), cov=covariance, size=n_samples
        )

    elif dist_name == "U[-sqrt(3), sqrt(3)]":
        bound = np.sqrt(3)

        return rng.uniform(low=-bound, high=bound, size=(n_samples, dim))

    elif dist_name == "t(5)":
        df = 5
        return rng.standard_t(df=df, size=(n_samples, dim)) * np.sqrt((df - 2) / df)

    elif dist_name == "Laplace(0, 1)":
        return rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2), size=(n_samples, dim))

    elif dist_name == "Dir(1, 1, 1)":
        return rng.dirichlet(
            alpha=np.ones(dim),
            size=n_samples,
        )

    elif dist_name == "Dir(10, 10, 10)":
        return rng.dirichlet(
            alpha=10 * np.ones(dim),
            size=n_samples,
        )
    else:
        raise ValueError(f"Unknown distribution: {dist_name!r}")
