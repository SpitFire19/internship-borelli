import numpy as np


def equal_gaussian_mixture(rng, dim, n_samples, separation=0.3):
    """
    Symmetric Gaussian mixture with

        E[X] = 0
        Cov(X) = I

    Parameters
    ----------
    rng : np.random.Generator
    dim : int
    n_samples : int
    separation : float
        ||mu||. Must satisfy separation < 1.
    """

    if separation >= 1:
        raise ValueError("Separation must be < 1")

    # mean vector
    mu = np.zeros(dim)
    mu[0] = separation
    # covariance ensuring Cov(X)=I
    Sigma = np.eye(dim) - np.outer(mu, mu)
    # randomly choose component
    labels = rng.integers(0, 2, size=n_samples)
    X = np.empty((n_samples, dim))
    idx = labels == 0
    X[idx] = rng.multivariate_normal(mu, Sigma, idx.sum())
    X[~idx] = rng.multivariate_normal(-mu, Sigma, (~idx).sum())
    return X


def gaussian_mixture_shifted(rng, dim, n_samples, mu, p=0.7):
    """
    Generate
        p * N(mu, I) + (1-p) * N(-mu, I)

    Parameters
    ----------
    rng : np.random.Generator
    dim : int
    n_samples : int
    mu : float or array-like
        Mean vector. If scalar, it is placed in the first coordinate.
    p : float
        Weight of N(mu,I).
    """

    # Build the mean vector
    if np.isscalar(mu):
        mean = np.zeros(dim)
        mean[0] = mu
    else:
        mean = np.asarray(mu)
        if len(mean) != dim:
            raise ValueError("mu must have length 'dim'")

    cov = np.eye(dim)
    # Component labels
    labels = rng.choice([0, 1], size=n_samples, p=[p, 1 - p])
    X = np.empty((n_samples, dim))
    idx = labels == 0
    X[idx] = rng.multivariate_normal(mean, cov, idx.sum())
    X[~idx] = rng.multivariate_normal(-mean, cov, (~idx).sum())
    return X


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

    elif dist_name == "Normal_mix_equal":
        return equal_gaussian_mixture(rng, dim, n_samples)

    elif dist_name == "Normal_mix_07":
        return gaussian_mixture_shifted(rng, dim, n_samples, [0.5, 0.3], 0.7)

    else:
        raise ValueError(f"Unknown distribution: {dist_name!r}")
