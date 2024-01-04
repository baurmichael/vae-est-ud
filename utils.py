import numpy as np
import torch
from typing import TypeVar

Tensor = TypeVar('Tensor')
ndarray = TypeVar('ndarray')


def rel_mse_np(x_true: Tensor, x_est: Tensor) -> Tensor:
    return np.sum(np.abs(x_true - x_est) ** 2, axis=-1) / np.mean(np.sum(np.abs(x_true) ** 2, axis=-1))


def dft_matrix(n: int, t='torch'):
    """
    Determines the DFT matrix of size n x n for given n
    :param n: number of row/columns in matrix
    :param t: return as tensor or numpy array
    :return: DFT matrix of size n x n
    """
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(-2 * np.pi * 1j / n)
    F = np.power(omega, i * j) / np.sqrt(n)
    return torch.tensor(F, dtype=torch.complex64) if t == 'torch' else F


def compute_lmmse(C_h: Tensor, mu: Tensor, y: Tensor, sigma, C_y=None, A=None, device='cpu') -> Tensor:
    B, N, M = C_h.shape[0], mu.shape[1], y.shape[1]

    # preprocess real and imaginary parts of components
    h_est = torch.zeros((B, N), dtype=torch.cfloat, device=device)

    # create identity matrix for A if it is None
    if A is None:
        A = torch.eye(M, dtype=torch.cfloat).to(device)

    for i in range(B):
        # compute LMMSE estimate for given observation and delta: h = mu + C_h*A^H (A*C_h*A^H + C_n)^-1 (y - A*mu)
        rhs = y[i] - torch.matmul(A, mu[i])
        if C_y is None:
            C_n = sigma[i]**2 * torch.eye(M, device=device)
            h_est[i] = mu[i] + C_h[i] @ A.H @ torch.linalg.solve(A @ C_h[i] @ A.H + C_n, rhs)
        else:
            h_est[i] = mu[i] + C_h[i] @ A.H @ torch.linalg.solve(C_y[i], rhs)
    return h_est


def genie_omp(A, D, y, h):
    """
    Apply the OMP algorithm to every row in y and h using the observation matrix A and dictionary D. The stopping
    condition is determined by evaluating the NMSE with respect to the true channel h.
    """
    AD = A @ D
    x_est = np.zeros((h.shape[0], D.shape[1]), dtype=h.dtype)

    for i in range(x_est.shape[0]):
        s = D.shape[1]
        x_prev = np.zeros(AD.shape[1], dtype=AD.dtype)
        x_curr = np.zeros(AD.shape[1], dtype=AD.dtype)
        S = np.zeros(AD.shape[1], dtype=bool)
        cost_curr = np.inf

        for _ in range(s):
            c = np.abs(AD.conj().T @ (y[i, :] - AD @ x_curr))
            c[S] = 0.0
            S[np.argmax(c)] = True
            x_curr.fill(0.0)
            x_curr[S] = np.linalg.lstsq(AD[:, S], y[i, :], rcond=-1)[0]

            # check if cost increased
            cost_prev = cost_curr
            cost_curr = (np.linalg.norm(D @ x_curr - h[i, :]) ** 2) / (np.linalg.norm(h[i, :]) ** 2)
            if cost_curr > cost_prev:
                break
            x_prev[:] = x_curr[:]

        x_est[i, :] = x_prev

    return x_est
