import numpy as np


def linear_phase(dt, T=1):
    phase = np.linspace(0, T, T / dt)[:, np.newaxis]
    
    return phase


def normalized_gaussian_basis(N, Z, dt):
    mu = np.linspace(0, 1, N)[:, np.newaxis]
    sigma = np.ones((N, 1)) / N
    
    basis = np.sqrt(2 * np.pi) * (1 / sigma.T) * np.exp(-0.5 * ((Z - mu.T) / sigma.T) ** 2)
    basis = basis / np.sum(basis, 1)[:, np.newaxis]

    basis = basis.T
    
    return basis
